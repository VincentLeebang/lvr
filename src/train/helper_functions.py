    '''Check trainable params'''
    # trainable_params = []
    # for name, param in model.named_parameters():
    #     # Check if the parameter is set to be trainable
    #     if param.requires_grad:
    #         trainable_params.append(name)
    #         print(f"- {name}")

    '''NaN checker hook to check where yields NaN'''
    # from functools import partial
    # nan_modules = {}

    # nan_layer_file = os.path.join("/dockerx/bangzhli/projects/LVR-Finetune/src/train/",f"nan_layers_detected_{training_args.run_name}.txt")
    # # re-write the log file
    # with open(nan_layer_file, 'w') as f:
    #     f.write("NaN Detection Log:\n" + "="*20 + "\n")
    # def nan_checker_hook(module, module_input, module_output, save_dir, nan_modules_dict):
    #     def _check_tensors(tensors, location):
    #         tensors_to_check = []
    #         if isinstance(tensors, tuple):
    #             tensors_to_check = list(tensors)
    #         elif isinstance(tensors, torch.Tensor):
    #             tensors_to_check.append(tensors)

    #         for i, tensor in enumerate(tensors_to_check):
    #             if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
    #                 # Find the name of the module this hook is attached to
    #                 module_name = ""
    #                 for name, m in model.named_modules():
    #                     if m is module:
    #                         module_name = name
    #                         break
                    
    #                 # We only want to save the tensors once per module to avoid clutter
    #                 if module_name not in nan_modules_dict:
    #                     nan_modules_dict[module_name] = True # Mark as found
                        
    #                     # print(f"!!! NaN Found in {location} of module: {module_name} !!!")
    #                     # print(f"!!! Saving input and output tensors to '{save_dir}' !!!")

    #                     # os.makedirs(save_dir, exist_ok=True)

    #                     # torch.save(module_input, os.path.join(save_dir, f"{module_name}_input.pt"))

    #                     # torch.save(module_output, os.path.join(save_dir, f"{module_name}_output.pt"))
                        
    #                     # Log to file
    #                     with open(nan_layer_file, 'a') as f:
    #                         f.write(f"NaN in {location} of {module_name}. Tensors saved.\n")

    #                 # We can stop checking after the first NaN is found and saved.
    #                 return

    #     # --- Main hook logic ---
    #     _check_tensors(module_input, "input")
    #     _check_tensors(module_output, "output")

    # for name, submodule in model.named_modules():
    #     hook_fn = partial(nan_checker_hook, save_dir="/dockerx/groups/bangzheng/nan_tensors", nan_modules_dict=nan_modules)
    #     submodule.register_forward_hook(hook_fn)

    '''End of forward hooks'''