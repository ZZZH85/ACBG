import os
backbone_path = './code/ACBG/resnet50-19c8e357.pth'
datasets_root='./datasets'
new_training_root = os.path.join(datasets_root, 'mix_casia2.0')
noau_training_root = os.path.join(datasets_root, 'CASIA2.0')
casia1_pth = './datasets/CASIAv1'
nist_pth = './datasets/NIST'
coverage_pth='./COVER/tampered'
columbia_pth='./Columbia/4cam_splc'
columbia_au_pth = './Columbia/4cam_auth'
coverage_all_root='./coverage/image'
imd_path = './datasets/IMD2020'
