from pathlib import Path

ref_dir = Path('/Users/dintu/zalo_ai/ZaloAi/mgc')
gen_dir = Path('/Users/dintu/zalo_ai/ZaloAi/syn_feat')

gFile = open('gen_files.list', 'w')
rFile = open('ref_files.list', 'w')

for file in gen_dir.glob('*.mgc'):
    ref_path = ref_dir.joinpath(file.name)
    if ref_path.is_file():
        gFile.write(f"{file}\n")
        rFile.write(f"{ref_path}\n")
    else:
        continue