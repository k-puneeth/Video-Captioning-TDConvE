import os
import torch
from acquire_images import sample_image
from encoder import ResTDconvE
from decoder import TDconvD
from train import get_sentence
from data_loader import sample_frame_util
import argparse
import json
from dataloader import you_cook2_dataset


def gen_caption(ckp_path, vid, vid_feat_path, return_top=1, beam_size=5, max_predict_length=2) :
	with open("youcookii_annotations_trainval.json","r") as f:
		data = json.load(f)["database"]

	segment = data[vid]["annotations"][0]["segment"]
	duration = data[vid]["duration"]


	dataloader = you_cook2_dataset("/vocab.txt","./data/feat_dat/train_frame_feat/","./youcookii_annotations_trainval",1,237)


    checkpoint = torch.load(ckp_path)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
	args.device=device
	encoder = ResTDconvE(args).to(args.device)
	decoder = TDconvD(args.embed_dim, args.decoder_dim,args.encoder_dim,args.attend_dim, len(w2i),args.device,layer=args.decoder_layer).to(args.device)

    encoder.load_state_dict(checkpoint['encoder'])
	decoder.load_state_dict(checkpoint['decoder'])
	encoder.eval()
	decoder.eval()

    # Get features here
    cur_frames = torchfile.load(vid_feat_path)
    cur_frames = sample_frame_util(cur_frames, duration, segment)
    features = encoder(cur_frames)
	(predict,prob)=decoder.predict(features,return_top=return_top,beam_size=beam_size,max_predict_length=max_predict_length)
	sentence = dataloader.decode_sentence(predict)
	print(sentence)

if _name_ == "_main_":
	gen_caption("./checkpoints_30/29th_ckp_66c49d81-ab0e-4d44-b5c7-966a6cf0cdc7","2Ihlw5FFrx4","./data/feat_dat/train_frame_feat/101/2Ihlw5FFrx4/0001/resnet_34_feat_mscoco.dat")
