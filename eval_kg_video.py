import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from utils.decord_func import decord_video_given_start_end_seconds
from utils.eval_utils import parse_choice, TypeAccuracy



QUESTION_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
                  'qa4_step','qa5_task', 'qa6_precedingStep', 'qa7_bestPrecedingStep',
                  'qa8_toolNextStep', 'qa9_bestInitial','qa10_bestFinal', 'qa11_domain']



def load_video(video_path, num_segments=8, start_secs=-1, end_secs=-1, return_msg=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    frame_indices = decord_video_given_start_end_seconds(video_path, 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=num_segments)

    frames = vr.get_batch(frame_indices).asnumpy()
    # print(frames.shape)
    if frames.shape[0] != num_segments:
        print('Concat frames...')
        frames = torch.from_numpy(frames)
        num_concat_frames = max(num_segments - frames.shape[0], 0)
        concat_frames = torch.zeros((num_concat_frames, frames.shape[1], frames.shape[2], frames.shape[3])).type_as(frames).to(frames.device)
        frames = torch.cat([frames, concat_frames], dim=0).numpy()

    frames = [Image.fromarray(v.astype('uint8')) for v in frames]

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames


def inference_video(model, tokenizer, question, vid_path, 
                    num_frame=16, start_secs=-1, end_secs=-1, params={}):
    frames, msg = load_video(
        vid_path, num_segments=num_frame, return_msg=True, 
        start_secs=start_secs, end_secs=end_secs
    )
    # print('start_secs', start_secs, 'end_secs', end_secs, msg)

    msgs = [
        {'role': 'user', 'content': frames + [question]}, 
    ]

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        **params
    )
    # print(answer)
    return answer



def main(args):
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Set decode params for video
    params = {}
    params["use_image_id"] = False
    params["max_slice_nums"] = args.max_slice_nums

    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Overall Accuracy for All Questions
    global_acc = TypeAccuracy("Global")
    qa_acc = []
    for t in range(len(QUESTION_TYPES)):
        qa_acc.append(TypeAccuracy(f"qa{t+1}_"))


    total = 0
    results = {}
    for line in tqdm(annotations, total=len(annotations)):
        # Q-A Pair
        idx = line["qid"]
        quest_type = line["quest_type"]
        conversations = line["conversations"]
        qs = conversations[0]["value"]
        gt_answers = conversations[1]["value"]
        results[idx] = {"qid": idx, "quest_type": quest_type, 
                        "qs": qs, "gt": gt_answers,
                        "task_label": line["task_label"], 
                        "step_label": line["step_label"]}
        qs = qs.replace("<video>\n", "")
        qs = qs.replace("<image>\n", "")
        vid_path = os.path.join(args.image_folder, line["video"])
        if "start_secs" in line:
            start_secs = line['start_secs']
            end_secs = line['end_secs']
        else:
            start_secs = -1
            end_secs = -1
        response = inference_video(model, tokenizer, 
                                   question=qs, 
                                   vid_path=vid_path, 
                                   num_frame=args.num_video_frames,
                                   start_secs=start_secs, end_secs=end_secs,
                                   params=params)

        total += 1
        answer_id = parse_choice(response, line["all_choices"], line["index2ans"])
        results[idx]["response"] = response
        results[idx]["parser"] = answer_id
        # print("qid {}:\n{}".format(idx, qs))
        # print("AI: {}\nParser: {}\nGT: {}\n".format(response, answer_id, gt_answers))

        global_acc.update(gt_answers, answer_id)
        for t in range(len(QUESTION_TYPES)):
            if f"qa{t+1}_" in quest_type:
                qa_acc[t].update(gt_answers, answer_id)

        # print each type accuracy
        print("-----"*5)
        acc_list = []
        for t in range(len(QUESTION_TYPES)):
            qa_acc[t].print_accuracy()
            acc_list.append(qa_acc[t].get_accuracy())
        global_acc.print_accuracy()
        print("-----"*5)
        avg_acc = sum(acc_list) / len(acc_list)
        print("Average Acc over Type: {:.4f}".format(avg_acc))

    # save all results
    print("save to {}".format(args.answers_file))
    with open(args.answers_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="openbmb/MiniCPM-V-2_6")
    parser.add_argument("--image-folder", type=str, default="data/COIN/videos")
    parser.add_argument("--question-file", type=str, default="data/testing_vqa20.json")
    parser.add_argument("--answers-file", type=str, default="data/answers_minicpmv_f8_q20.json")
    parser.add_argument("--max_slice_nums", type=int, default=2, help="use 1 if cuda OOM and video resolution > 448*448")
    # parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_video_frames", type=int, default=8)
    args = parser.parse_args()
    main(args)

