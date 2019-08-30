from pathlib import Path

ref_dir = Path('/Users/dintu/zalo_ai/postfilt_gan/natural_mgc')
gen_dir = Path('/Users/dintu/zalo_ai/postfilt_gan/synthesized_mgc')

# ref_dir = Path('/Training/tdinh/postfilt_gan/mgc')
# gen_dir = Path('/Training/tdinh/postfilt_gan/mgc_gen')

gFile = open('gen_files.list', 'w')
rFile = open('ref_files.list', 'w')
count = 0
for file in gen_dir.glob('*.mgc'):
    ref_path = ref_dir.joinpath(file.name)
    if ref_path.is_file():
        gFile.write(f"{file}\n")
        rFile.write(f"{ref_path}\n")
        count += 1
    else:
        continue
print(count)