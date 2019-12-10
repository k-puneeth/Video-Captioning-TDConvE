from torch.utils.data import Dataset, DataLoader
import json
import os
# from skimage import io
import torch
import torchfile
import string
import numpy as np

g_sample=10

def sample_frame_util(frames, duration, segment):

    ratio = duration / 500.0
    start = int(float(segment[0]) / ratio)
    end = int(float(segment[1]) / ratio)

    interval = int((end - start) / g_sample)

    l = [start + interval * i  for i in range(10)]

    #check
    if len(l) != 10 :
        print("PROBLEM!!")

    return frames[l]


class you_cook2_dataset(Dataset):
    def __init__(self,vocab_file,dataset_dir,meta_file,batch,max_len):

        self.batch=batch
        self.dataset_dir = dataset_dir
        self.max_len = max_len

        # reading vocabulary
        self.table = str.maketrans('', '', string.punctuation)
        with open(vocab_file, "r") as f :
            self.vocab = [w.strip() for w in f.readlines()]
            self.w2i = {}
            self.i2w = {}

        for idx, w in enumerate(self.vocab + ["<unk>", "<sos>", "<eos>", "<pad>"]):
            self.w2i[w] = idx
            self.i2w[idx] = w

        #making video list
        dirs = os.listdir(dataset_dir)
        self.vid_paths = {}
        for dir in dirs :
            # self.vids.append([(dir, i) for i in os.listdir(dataset_dir + "/" dir)])
            for i in os.listdir(dataset_dir + "/" + dir) :
               self.vid_paths[i] = dataset_dir + "/" + dir + "/" + i + "/" #+ "0001/resnet_34_feat_mscoco.dat"

        self.meta = json.load(open(meta_file, "r"))['database']
        self.clips = []
        for key in self.meta.keys() :
            if key in self.vid_paths.keys() :
                for i in range(3):
                    if i == 0 :
                        addon = "0001/resnet_34_feat_mscoco.dat"
                    if i == 1 :
                        addon = "0002/resnet_34_feat_mscoco.dat"
                    if i == 2 :
                        addon = "0003/resnet_34_feat_mscoco.dat"
                    if (not os.path.exists(self.vid_paths[key] + addon)) : continue
                    md = {"vid" : key, "duration" : self.meta[key]['duration'], "dup":i}
                    self.clips.extend([(md, i) for i in self.meta[key]['annotations']])

    def __getitem__(self,idx):


        cur_clips = self.clips[idx * self.batch : (idx * self.batch) + self.batch]
        frames = []
        sen = []
        lens = []
        for meta, seg in cur_clips :
            # print(meta, seg)
            if meta["dup"] == 0 :
                addon = "0001/resnet_34_feat_mscoco.dat"
            if meta["dup"] == 1 :
                addon = "0002/resnet_34_feat_mscoco.dat"
            if meta["dup"] == 2 :
                addon = "0003/resnet_34_feat_mscoco.dat"
            cur_frames = torchfile.load(self.vid_paths[meta["vid"]] + addon)
            cur_frames = sample_frame_util(cur_frames, meta['duration'], seg['segment'])
            if cur_frames.shape[0] != 10 :
                print("PROBLEM!!!")
            cur_sen, cur_len = self.encode(seg['sentence'])
            frames.append(cur_frames)
            sen.append(cur_sen)
            lens.append(cur_len)

        # returns (batch * g_sample * 512), (g_sample, max_len), (max_len)
        return torch.from_numpy(np.array(frames)), torch.from_numpy(np.array(sen)), torch.from_numpy(np.array(lens))
        # return np.array(frames), np.array(sen), np.array(lens)


    def __len__(self):
        return int(len(self.clips) / self.batch)

    def encode(self, s):

        s = self.clean_doc(s)
        s = ["<sos>"] + s + ["<eos>"]
        s = [self.w2i.get(tok, self.w2i["<unk>"]) for tok in s]
        return np.pad(np.array(s), (0, self.max_len - len(s)), "constant", constant_values = self.w2i["<pad>"]), len(s)

    def clean_doc(self, doc):

        tokens = doc.split()
        # remove punctuation from each token
        tokens = [w.translate(self.table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # make lower case
        tokens = [word.lower() for word in tokens]
        return tokens

    def decode_sentence(self, sen_in):
    	#sen_in is 2d array of size batch*sentence_len. sentences includes <pad>, <sos>,<eos>,<unk> etc.
    	sen=[]
    	for s in sen_in:
    		temp=[]
    		for i in s:
    			i=i.item()
    			if i==self.w2i["<eos>"]:
    				break
    			temp.append(i)
    		#temp can be of length 1 to max_predict now. First one is always <sos>. There won't be <eos> in it now.
    		sen.append([self.i2w[i] for i in temp if (i!=self.w2i["<sos>"] and i!=self.w2i['<pad>'])])

    	#sen is a list(len=batch) of list(len=# of words in this sentence, from 0 to max_predict-1) of string.
    	return sen



# if __name__=="__main__":
# 	msr_vtt=msr_vtt_dataset("/tmp3/data/","train",17)
# 	print(msr_vtt[5])
# 	print(msr_vtt[-1])
