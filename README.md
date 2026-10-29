# ATHENA Spectral API — organizer-only deployment

Keep this repository **private**. The Python simulator defines the hidden phase diagram, clean spectra, and noise model.

## Deploy on Render

1. Put the files in this folder at the root of a private GitHub repository.
2. In Render choose **New > Blueprint**, connect the repository, and deploy its `render.yaml`.
3. Copy the actual public HTTPS service URL from the Render dashboard.
4. Test `https://<your-service>.onrender.com/info`.
5. Set `BASE_URL` in participant notebooks to your actual URL, not a placeholder.

## Important limitations before a scored competition

- `/spectrum` currently exposes a full noisy spectrum in one request, bypassing the intended point-by-point challenge; disable it for the competition.
- `/submit` returns the RMSE immediately, enabling repeated feedback across new sessions.
- `/session` can be called repeatedly with no participant authentication or episode limit.
- Sessions are in process memory, so restarts lose them; use persistent storage for real competition use.
- Add rate limiting and identity controls; this is a local prototype, not a hardened public competition backend.
- Free-plan availability and plan constraints may vary; verify on Render before deploying.
