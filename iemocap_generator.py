import numpy as np
import os

def _get_data(file_path):
    # print(csv_path)
    print(file_path)
    f = open(file_path)
    valance, arousal, timestamp = [], [], []
    line = f.readline()
    time = 0

    while line:
        if line.startswith('['):
            sq = line.split()[3]

            # print(line.split)
            val = line.split()[5][1:7]
            aro = line.split()[6][0:6]
            val = (float(val) - 3) / 2
            aro = (float(aro) - 3) / 2

            # print(float(val), float(aro))
            end = float(line.split()[2][:-1])
            start = float(line.split()[0][1:])
            dur = float(end) - float(start)
            # print(dur)
            # print("time = ", time)
            # print("start = ", start)

            # print("end = ", end)
            print(line)

            for t in range(time, int(start)):
                timestamp.append(t)
                valance.append(0.00)
                arousal.append(0.00)

            if (int(start) > time):
                time = int(start)

            for t in range(time, int(end) + 1):
                timestamp.append(t)
                valance.append(val)
                arousal.append(aro)

            time = int(end) + 1

        line = f.readline()
    timestamp = np.array(timestamp).reshape(-1, 1)
    valance = np.array(valance).reshape(-1, 1)
    arousal = np.array(arousal).reshape(-1, 1)
    # print(valance.shape)
    # print(arousal.shape)
    f.close()

    return timestamp.astype(np.float32), arousal.astype(np.float32), valance.astype(np.float32)


modality_dir = "./iemocap_data"
mod_file = './iemocap_data/Ses01F_impro01.wav'
# for mod_file in os.listdir(modality_dir + "/Ses01F_impro01"):
# if (mod_file[-4:] != '.wav'):
#    continue
sq = mod_file[:-4]
# print(sq)

arousal_label_path = "./iemocap_data/Ses01F_impro01.txt"

timestamp, arousal_ratings, valence_ratings = _get_data(arousal_label_path)

data = np.hstack([timestamp, arousal_ratings, valence_ratings])
label_file = './iemocap_data/labels/' + ('Ses01F_impro01' + '.csv')

np.savetxt(str(label_file), data, header='timestamp,arousal,valence', fmt='%f', delimiter=',')


files = []


# for mod_file in os.listdir(modality_dir + "/Ses01F_impro01"):
  # if (mod_file[-4:] != '.wav'):
    # continue
  # audio_file = os.path.join(modality_dir, 'Ses01F_impro01', mod_file)
  # label_file = os.path.join(modality_dir,'labels', mod_file[:-4] + '.csv')
label_file = "./iemocap_data/labels/Ses01F_impro01.csv"
audio_file = "./iemocap_data/Ses01F_impro01.wav"

files.append([str(audio_file), str(label_file)])

save_inp_file = modality_dir + '/input_file.csv'
print(save_inp_file)

np.savetxt(str(save_inp_file), np.array(files), header='file,label_file', fmt='%s', delimiter=',')

from end2you.data_generator import FileReader, AudioGenerator, VisualGenerator
from pathlib import Path

path_to_save_hdf5 = Path('./iemocap_data/hdf5')
input_file_path = Path('./iemocap_data/input_file.csv')

filereader = FileReader(',')
labelfile_reader = FileReader(delimiter=',')
visual_generator = AudioGenerator(save_data_folder=str(path_to_save_hdf5),
                                 input_file=str(input_file_path),
                                 reader=filereader,
                                 labelfile_reader=labelfile_reader)
visual_generator.write_data_files()


