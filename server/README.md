# Self-hosting on Proxmox

Runs the whole pipeline — data fetch, model tournament, ledger, line
collection, dashboard — on your own hardware, with an API for triggering
runs and reading status. GitHub stays the source of truth: with
`GIT_SYNC=1` every job pulls before running and pushes results after,
so the GitHub Actions remain a fallback and offsite backup.

## Proxmox setup (Debian 12 LXC, ~10 minutes)

1. **Create the container** (Proxmox UI or shell):
   - Template: `debian-12-standard`, 2 vCPU / 2 GB RAM / 16 GB disk
   - In Options → Features, enable `nesting=1` (required for Docker in LXC)

2. **Inside the container:**
   ```bash
   apt update && apt install -y curl git
   curl -fsSL https://get.docker.com | sh

   git clone https://github.com/jhaberhern/jhaberhern.git /opt/nfl-model
   cd /opt/nfl-model/server
   echo "ODDS_API_KEY=your-key-here" > .env
   docker compose up -d --build
   ```

3. **Open the UI:** `http://<container-ip>:8000` — same dashboard,
   now served locally, plus:
   - `GET  /api/status` — every job's schedule, last run, next run
   - `POST /api/run/{fetch|improve|ledger|collect|export}` — run now
   - `GET  /api/log/{job}` — the last run's full output

## Pushing results back to GitHub (recommended)

Create a fine-grained PAT (repo → Settings → Developer settings) with
*Contents: read/write* on this repo only, then:

```bash
cd /opt/nfl-model
git remote set-url origin https://<PAT>@github.com/jhaberhern/jhaberhern.git
echo "GIT_SYNC=1" >> server/.env
docker compose -f server/docker-compose.yml up -d
```

If you go this route, **disable the schedule triggers in the GitHub
workflows** (delete the `schedule:` blocks, keep `workflow_dispatch`) so
your server and GitHub's cron don't race each other and create
conflicting commits. GitHub Actions then remains available as a manual
fallback if the server is ever down.

## Remote access from your phone

Install [Tailscale](https://tailscale.com) in the container
(`curl -fsSL https://tailscale.com/install.sh | sh && tailscale up`) and
the dashboard is reachable at `http://<tailscale-ip>:8000` from
anywhere, with zero exposed ports. Do not port-forward this from the
open internet — there is no authentication on the API.

## Changing the schedule

Job commands and cron strings live at the top of `server/app.py`
(`JOBS`). Want daily retraining instead of weekly? Change the improve
job's cron to `0 12 * * *` and `docker compose restart`. More frequent
line snapshots near kickoff are the highest-value upgrade — each
collector run costs ~3 API credits against the free tier's 500/month.
