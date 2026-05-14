# WRDS setup

How to get your local environment talking to Wharton Research Data Services so `python -m src.data.ingest_wrds` works.

## 1. Confirm you have a WRDS account

Your school must have a WRDS subscription, and you need an active individual account.

1. Go to https://wrds-www.wharton.upenn.edu/
2. Log in (usually via your school SSO).
3. If you can browse the data dictionaries, your account is active.

If login fails or you don't have an account, contact your school's library / research support.

## 2. Find your real WRDS username

**Your WRDS username is NOT necessarily your school SSO username.** It's separately assigned by WRDS.

1. Logged in at wrds-www.wharton.upenn.edu, click your name (top right) → **My Account**.
2. The **Username** field on that page is the one to use everywhere.
3. Save it in your project `.env`:
   ```
   WRDS_USERNAME=<your_actual_wrds_username>
   ```

## 3. Set a WRDS PostgreSQL password

PostgreSQL access uses a password you set explicitly — it is NOT your SSO password.

1. On the WRDS site → **My Account** → **Change Password**.
2. Set a new password. Use only ASCII characters; avoid `'`, `"`, `\`, and `:` (they break `.pgpass` parsing).

## 4. First-time `wrds` Python setup

Install the package (already in `requirements.txt`):
```bash
pip install -e .
# or just:
pip install wrds psycopg2-binary
```

Run the interactive setup once:
```bash
python -c "import wrds; wrds.Connection()"
```

It prompts:
- WRDS username (use the one from step 2)
- WRDS password (the one from step 3)
- "Would you like to create a .pgpass file?" → answer **`y`**.

This writes `~/.pgpass` with mode 0600 containing the credentials. Future `wrds.Connection()` calls authenticate automatically with no prompts.

## 5. Verify

```bash
python -c "
import wrds
conn = wrds.Connection(wrds_username='YOUR_USERNAME')
print(conn.list_libraries()[:5])
conn.close()
"
```

Should print 5 library names (e.g. `['audit', 'bank', 'block', 'boardex', 'cboe']`).

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `FATAL: password authentication failed for user "X"` | Wrong username, or password mismatch | Re-check step 2 (real username); re-set password in step 3. |
| `FATAL: role "X" does not exist` | Username is your school SSO, not your WRDS username | See step 2. |
| `could not translate host name` / connection refused | DNS / firewall blocking `wrds-pgdata.wharton.upenn.edu:9737` | On VPN? Try off VPN. Otherwise check school's network. |
| `Permission denied: '~/.pgpass'` | File mode not 0600 | `chmod 600 ~/.pgpass` |
| `ImportError: No module named 'wrds'` | Package not installed | `pip install wrds psycopg2-binary` |
| `ModuleNotFoundError: psycopg2` | psycopg2 missing | `pip install psycopg2-binary` |

After step 5 works, run a pull:
```bash
python -m src.data.ingest_wrds --linktable-only       # smallest, fastest probe
python -m src.data.ingest_wrds --all                  # full pull
```
