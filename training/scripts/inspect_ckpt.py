import torch
ckpt_path='training/checkpoints/best_model.pth'
ck = torch.load(ckpt_path, map_location='cpu')
print('species_mapping type:', type(ck.get('species_mapping')))
idm = ck.get('species_mapping', {}).get('id_to_species')
print('id_to_species type:', type(idm))
if idm:
    keys=list(idm.keys())
    print('sample keys:', keys[:5])
    print('key types:', [type(k) for k in keys[:5]])
else:
    print('id_to_species not found')
