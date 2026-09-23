# ops/ — the authors' training-host scripts

Not needed to use or evaluate the model. `remote.sh` syncs this repo to a GPU host over ssh/rsync
(`REMOTE=user@host ops/remote.sh sync|setup|train …`), `remote_setup.sh` builds the host environment,
`run_queue.sh` trains and evaluates a job list sequentially, `jobs/` are the queues used for the
experiments documented in docs/RESULTS_*.md.
