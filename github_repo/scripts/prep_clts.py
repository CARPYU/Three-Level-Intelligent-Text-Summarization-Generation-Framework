# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path

def convert_pair(src_file: Path, tgt_file: Path, out_file: Path):
    with src_file.open("r", encoding="utf-8") as fs, \
         tgt_file.open("r", encoding="utf-8") as ft, \
         out_file.open("w", encoding="utf-8") as fo:
        for src, tgt in zip(fs, ft):
            src = src.strip()
            tgt = tgt.strip()
            # 若数据里有 <q>/<sep> 等分隔符，可统一替换
            for sep in ["<q>", "<n>", "<sep>", "##"]:
                src = src.replace(sep, "\n")
                tgt = tgt.replace(sep, "。")
            ex = {
                "messages": [
                    {"role": "system", "content": "你是资深中文新闻编辑，擅长在100-200字内写忠实、凝练的摘要。"},
                    {"role": "user", "content": f"请为下面的新闻生成中文摘要，避免凭空捏造。\n\n【新闻】{src}"},
                    {"role": "assistant", "content": tgt}
                ]
            }
            fo.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="dir containing *.src/*.tgt")
    ap.add_argument("--out_dir", required=True, help="dir to write *_chat.jsonl")
    args = ap.parse_args()

    raw = Path(args.raw_dir); out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = [("train.src","train.tgt","train_chat.jsonl"),
             ("valid.src","valid.tgt","valid_chat.jsonl"),
             ("test.src","test.tgt","test_chat.jsonl")]
    for src, tgt, outname in pairs:
        src_p, tgt_p = raw/src, raw/tgt
        if not (src_p.exists() and tgt_p.exists()):
            print(f"[Skip] missing {src_p} or {tgt_p}")
            continue
        print(f"[Convert] {src} + {tgt} -> {outname}")
        convert_pair(src_p, tgt_p, out/outname)

if __name__ == "__main__":
    main()

