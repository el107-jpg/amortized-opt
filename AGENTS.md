# Project instructions

## Training dashboards

Whenever launching a training run, enable TensorBoard output, register its
event directory under `runs/tensorboard_active`, and verify that the shared
TensorBoard server on port 6006 lists the run. Keep existing registered runs
visible unless the user explicitly asks to remove them. Disposable startup
smokes may opt out with `TB_AUTO_REGISTER=0`.

For all training runs, we are using the critic with bellman on.