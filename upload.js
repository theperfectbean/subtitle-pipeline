#!/usr/bin/env node
'use strict';
/**
 * upload.js
 * CLI entry point for uploading subtitles to OpenSubtitles.
 *
 * Usage:
 *   node upload.js --show shows/pumuckl-1982.yaml all
 *   node upload.js --show shows/pumuckl-1982.yaml S02
 *   node upload.js --show shows/pumuckl-1982.yaml S01E01
 *   node upload.js --show shows/pumuckl-1982.yaml --dry-run all
 */

const fs   = require('fs');
const path = require('path');
const os   = require('os');
const { spawnSync } = require('child_process');

require('dotenv').config({ path: path.join(__dirname, '.env') });
const yaml = require('js-yaml');
const OS   = require('opensubtitles-api');

// ── Logging ───────────────────────────────────────────────────────────────────

let _logFile = null;

function log(level, msg) {
    const line = `${new Date().toISOString()} [${level}] ${msg}`;
    process.stderr.write(line + '\n');
    if (_logFile) fs.appendFileSync(_logFile, line + '\n', { encoding: 'utf8' });
}
const info  = (m) => log('INFO',  m);
const warn  = (m) => log('WARN',  m);
const error = (m) => log('ERROR', m);

// ── Config ────────────────────────────────────────────────────────────────────

function deriveSlug(name) {
    return name.replace(/[()]/g, '').trim().toLowerCase()
        .replace(/\s+/g, '-').replace(/-+/g, '-');
}

function loadShow(yamlPath) {
    if (!fs.existsSync(yamlPath)) {
        error(`Show config not found: ${yamlPath}`);
        process.exit(1);
    }
    const raw = yaml.load(fs.readFileSync(yamlPath, 'utf8'));
    for (const field of ['name', 'media_dir', 'opensubtitles_imdb_id']) {
        if (!raw[field]) {
            error(`Show config ${yamlPath} is missing required field: ${field}`);
            process.exit(1);
        }
    }
    const showSlug = deriveSlug(raw.name);
    fs.mkdirSync('/home/admin/logs', { recursive: true });
    return {
        name:        raw.name,
        media_dir:   raw.media_dir,
        media_host:  raw.media_host  || '192.168.0.113',
        media_user:  raw.media_user  || 'admin',
        source_lang: raw.source_lang || 'de',
        target_lang: raw.target_lang || 'en',
        imdb_id:     String(raw.opensubtitles_imdb_id),
        show_slug:   showSlug,
        upload_log:  `/home/admin/logs/subtitle-pipeline-${showSlug}-upload.log`,
    };
}

// ── Argument Parsing ──────────────────────────────────────────────────────────

function parseArgs() {
    const args = process.argv.slice(2);
    let showPath = null, target = null, dryRun = false;
    for (let i = 0; i < args.length; i++) {
        if      (args[i] === '--show')    showPath = args[++i];
        else if (args[i] === '--dry-run') dryRun   = true;
        else                              target   = args[i];
    }
    if (!showPath || !target) {
        process.stderr.write(
            'Usage: node upload.js --show shows/<show>.yaml [--dry-run] <all|SxxExx|Sxx>\n'
        );
        process.exit(1);
    }
    return { showPath, target, dryRun };
}

// ── Target Matching ───────────────────────────────────────────────────────────

function matchesTarget(epId, target) {
    if (target === 'all')            return true;
    if (/^S\d{2}E\d{2}$/.test(target)) return epId === target;
    if (/^S\d{2}$/.test(target))    return epId.startsWith(target);
    error(`Unrecognized target format: ${target}`);
    process.exit(1);
}

// ── SSH / SCP ─────────────────────────────────────────────────────────────────

const SSH_BASE = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                  '-i', '/home/admin/.ssh/id_rsa'];

function runSsh(host, user, cmd) {
    const r = spawnSync('timeout', ['60', 'ssh', ...SSH_BASE, `${user}@${host}`, cmd],
        { encoding: 'utf8' });
    if (r.error) throw new Error(`SSH spawn error: ${r.error.message}`);
    return r;
}

function scpFrom(host, user, remotePath, localPath) {
    const r = spawnSync('scp', ['-q', ...SSH_BASE, `${user}@${host}:${remotePath}`, localPath],
        { encoding: 'utf8' });
    if (r.error || r.status !== 0) {
        throw new Error(`SCP failed for ${path.basename(remotePath)}: ${r.stderr.trim()}`);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
    const { showPath, target, dryRun } = parseArgs();
    const cfg = loadShow(showPath);
    _logFile = cfg.upload_log;

    info(`=== Upload Run: show=${cfg.name}, target=${target}, DRY_RUN=${dryRun} ===`);

    const osUser = process.env.OS_USERNAME;
    const osPass = process.env.OS_PASSWORD;
    if (!osUser || !osPass) {
        error('OS_USERNAME and OS_PASSWORD must be set in environment or .env');
        process.exit(1);
    }

    // Verify SSH connectivity
    const connTest = runSsh(cfg.media_host, cfg.media_user, 'true');
    if (connTest.status !== 0) {
        error(`Unable to connect to ${cfg.media_user}@${cfg.media_host} via SSH`);
        process.exit(1);
    }

    // Find source-language SRT files to identify episode IDs
    const findRes = runSsh(cfg.media_host, cfg.media_user,
        `find '${cfg.media_dir}' -name '*.${cfg.source_lang}.srt' | sort`);
    if (findRes.status !== 0) {
        error(`Failed to list source SRT files on ${cfg.media_host}: ${findRes.stderr.trim()}`);
        process.exit(1);
    }
    const srtPaths = findRes.stdout.split('\n').map(l => l.trim()).filter(Boolean);

    const OpenSubtitles = new OS({
        useragent: 'SubtitlePipeline',
        username:  osUser,
        password:  osPass,
        ssl:       true,
    });

    let loggedIn   = false;
    let found      = 0;
    let processed  = 0;

    for (const deSrtPath of srtPaths) {
        const epMatch = deSrtPath.match(/S\d{2}E\d{2}/);
        if (!epMatch) continue;
        const epId = epMatch[0];
        if (!matchesTarget(epId, target)) continue;

        found++;

        if (dryRun) {
            info(`DRY RUN match: ${deSrtPath} (${epId})`);
            continue;
        }

        info(`--- ${epId} ---`);

        // Locate companion files
        const enSrtRes = runSsh(cfg.media_host, cfg.media_user,
            `find '${cfg.media_dir}' -name '*${epId}*.${cfg.target_lang}.srt' 2>/dev/null | head -1`);
        const enSrtPath = enSrtRes.stdout.trim();

        const mkvRes = runSsh(cfg.media_host, cfg.media_user,
            `find '${cfg.media_dir}' -name '*${epId}*.mkv' 2>/dev/null | head -1`);
        const mkvPath = mkvRes.stdout.trim();

        if (!mkvPath) {
            warn(`${epId}: no MKV found — skipping`);
            continue;
        }

        const tmpDir = path.join(os.tmpdir(), `os-upload-${cfg.show_slug}-${epId}`);
        fs.mkdirSync(tmpDir, { recursive: true });

        const localMkv   = path.join(tmpDir, path.basename(mkvPath));
        const localDeSrt = path.join(tmpDir, path.basename(deSrtPath));
        const localEnSrt = enSrtPath ? path.join(tmpDir, path.basename(enSrtPath)) : null;

        try {
            info(`${epId}: downloading to ${tmpDir}`);
            scpFrom(cfg.media_host, cfg.media_user, mkvPath,   localMkv);
            scpFrom(cfg.media_host, cfg.media_user, deSrtPath, localDeSrt);
            if (enSrtPath) {
                scpFrom(cfg.media_host, cfg.media_user, enSrtPath, localEnSrt);
            }

            if (!loggedIn) {
                info('Logging in to OpenSubtitles...');
                await OpenSubtitles.login();
                loggedIn = true;
                info('Login successful');
            }

            // Upload source-language SRT
            info(`${epId}: uploading ${path.basename(localDeSrt)}`);
            const deResult = await OpenSubtitles.upload({
                path:    localMkv,
                subpath: localDeSrt,
                imdbid:  cfg.imdb_id,
            });
            info(`${epId}: DE result: ${JSON.stringify(deResult)}`);

            // Upload target-language SRT if present
            if (localEnSrt) {
                info(`${epId}: uploading ${path.basename(localEnSrt)}`);
                const enResult = await OpenSubtitles.upload({
                    path:    localMkv,
                    subpath: localEnSrt,
                    imdbid:  cfg.imdb_id,
                });
                info(`${epId}: EN result: ${JSON.stringify(enResult)}`);
            }

            processed++;
        } catch (err) {
            error(`${epId}: upload failed — ${err.message}`);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
            info(`${epId}: temp dir cleaned up`);
        }
    }

    if (found === 0) {
        warn(`No matching episodes found (target=${target})`);
    } else if (dryRun) {
        info(`DRY RUN complete — ${found} matching file(s) found, no upload performed.`);
    } else {
        info(`Upload run complete — ${processed}/${found} episode(s) processed.`);
    }

    info('=== Run complete ===');
}

main().catch(err => {
    error(`Fatal: ${err.message}`);
    process.exit(1);
});
