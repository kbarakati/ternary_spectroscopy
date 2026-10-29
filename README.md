# ATHENA two-challenge private API

Deploy these files to a **private** GitHub repository connected to Render. `render.yaml` configures a single Python web worker. The public participant notebook is `ATHENA_Ternary_Spectroscopy_Both_Challenges.ipynb` (distribute separately).

API: `GET /info`; `POST /session` with `{ "challenge": 1 }` or `{ "challenge": 2 }`; `POST /pixel` with `X-Session-ID` header and `{ "wavelength":4.5,"time":1.0 }` for Challenge 1, or add `{ "x":0.3,"y":0.4 }` for Challenge 2; `GET /budget`; `POST /submit` (Challenge 1); `POST /submit-cube` (Challenge 2). `/spectrum` is disabled to enforce sparse-channel sampling.

For Challenge 2, `GET /info` supplies ordered `cube_grid` coordinates. The submitted `predictions` matrix must have shape `(len(cube_grid), 512)`, in precisely this row order. Compositions use x>=0, y>=0, x+y<=1. Each task has its own acquisition-time budget.

Important: This is a prototype and is **not production-secure for public competitive scoring**. Sessions are in volatile process memory; they have no rate limits or per-team quotas and scores are returned after submission. Add persistence, auth and a private leaderboard before scored public use. The instructor simulator is proprietary/private benchmark code: do not publish its repository or share its Python file with participants.
