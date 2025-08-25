from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/unianimate', cache_dir='checkpoints/')


from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('iic/tf-t2v', cache_dir='models/')