(qrnn3d) fy@pc src $ python -m matpy.runner   --clean-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/mat   --out-dir /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois   --model-path /mnt/data/User/HSID/Pre_Model/qrnn3d/gauss/model_epoch_50_118454.pth   --noise-type gaussian   --sigma 25.0   --seed 0   --result-json /mnt/data/User/HSID/Dataset/ICVL-BGU/ICVL_HS_2016/nois/results.json   --action full
Traceback (most recent call last):
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/matpy/runner.py", line 149, in <module>
    main()
  File "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/matpy/runner.py", line 143, in main
    summary = full_pipeline(args.clean_dir, args.out_dir, args.model_path, args.noise_type, args.sigma, args.seed, args.result_json)
  File "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/matpy/runner.py", line 103, in full_pipeline
    predict_dataset(noisy_dir, model_path, pred_dir)
  File "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/matpy/runner.py", line 60, in predict_dataset
    pred = infer_on_hsi(model, noisy, device)
  File "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/matpy/model_eval.py", line 70, in infer_on_hsi
    out = model(tensor)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/models/qrnn/utils.py", line 32, in forward
    out = self.feature_extractor(xs[0])
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/models/qrnn/qrnn3d.py", line 81, in forward
    Z, F1, F2 = self._conv_step(inputs)
  File "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/models/qrnn/qrnn3d.py", line 68, in _conv_step
    gates = self.conv(inputs)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/container.py", line 240, in forward
    input = module(input)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 725, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/home/fy/miniforge3/envs/qrnn3d/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 720, in _conv_forward
    return F.conv3d(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 10.03 GiB. GPU 0 has a total capacity of 7.92 GiB of which 7.36 GiB is free. Including non-PyTorch memory, this process has 554.00 MiB memory in use. Of the allocated memory 431.31 MiB is allocated by PyTorch, and 706.50 KiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
(qrnn3d) fy@pc src $ 