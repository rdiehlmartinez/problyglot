__author__ = 'Richard Diehl Martinez'
''' Utility functions related to data processing functionality '''

def move_to_device(batch, device):
    updated_batch = {}
    for key, val in batch.items():
        if isinstance(val, dict):
            if key not in updated_batch:
                updated_batch[key] = {}
            for sub_key, sub_val in val.items():
                if sub_val is not None:
                    updated_batch[key][sub_key] = sub_val.to(device)
        else:
            if val is not None:
                updated_batch[key] = val.to(device)
    return updated_batch