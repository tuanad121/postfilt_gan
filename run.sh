# For training 
# python main_1.py --voiceName f1 --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9999 --mode train --cuda

#or

nohup python main.py --voiceName f1 --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9999 --mode train --niter 50 --beta1 0.5 --workers 10 &

