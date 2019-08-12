# For training 
# python main_1.py --voiceName f1 --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9999 --mode train --cuda

#or

nohup nice -n 19 python main.py --voiceName f1 --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9999 --mode train --workers 10 --cuda &

