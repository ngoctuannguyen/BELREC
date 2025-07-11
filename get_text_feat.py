import argparse
import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import ast

def arg_parse():
    parser = argparse.ArgumentParser(description="Get text features a dataset.")
    parser.add_argument("--dataset", type=str, required=True, 
                        default="baby", help="Name of the dataset.")
    parser.add_argument("--text_column", type=str, required=True, 
                        default="title", help="Name of the column containing text data.")
    parser.add_argument("--txt_embedding_model", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Name of the text embedding model to use.")
    args = parser.parse_args()
    return args

def get_encoding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

def get_dataset(args) -> dict: 
    dataset_folder = {
        "inter_file": f"data/{args.dataset}/{args.dataset}.inter",
        "mapping_file": f"data/{args.dataset}/i_id_mapping.csv",
        "description_file": f"data/{args.dataset}/amazon_descriptions_{args.dataset}_sample.json",
        "meta_data": f"data/{args.dataset}/meta_{args.dataset}.json",
        "u_id_mapping": "data/{args.dataset}/u_id_mapping.csv",
        "five_core_data": f"data/{args.dataset}/{args.dataset}_5.json",
    }
    return dataset_folder

def get_details_dataset_df(args, mapping_file, five_core_data, description_file, text_column) -> pd.DataFrame:
    
    with open(os.path.join(description_file), "r") as f:
        description_data = json.load(f)

    description_df = pd.DataFrame(description_data)

    i_id_mapping = pd.read_csv(os.path.join(mapping_file), sep="\t")

    ### map item with description sample on asin 
    item_id_with_description_df = i_id_mapping.merge(description_df, on="asin", how="left")
    ### get five core data to fill missing descriptions
    baby_five_core_data = []
    with open(os.path.join(five_core_data), "r") as f:
        for line in f:
            baby_five_core_data.append(json.loads(line))

    baby_df = pd.DataFrame(baby_five_core_data)

    ### merge with five core data
    item_id_with_description_df = item_id_with_description_df.merge(baby_df, on="asin", how="left")
    for index, row in item_id_with_description_df.iterrows():
        if pd.isnull(row[text_column]):
            for i in baby_five_core_data:
                if i["asin"] == row["asin"]:
                    item_id_with_description_df.loc[index, text_column] = i["reviewText"]
                    break
    
    return item_id_with_description_df

def concat_with_meta_data(args, df, meta_data_file, text_column):
    """
    Concatenate the DataFrame with metadata from the specified file.
    """
    meta_data = []
    with open(os.path.join(meta_data_file), "r") as f:
        for line in f:
            # Safely evaluate the string literal before loading as JSON
            meta_data.append(json.loads(json.dumps(ast.literal_eval(line.strip()))))

    meta_df = pd.DataFrame(meta_data)
    meta_df_5_core = meta_df.merge(df, on="asin", how="left")
    meta_df_5_core.rename(columns={"title_x": "title"}, inplace=True)

    ### Concat
    for id, row in enumerate(meta_df_5_core["description"]):
      if pd.isna(row):
        meta_df_5_core.loc[id, "description"] = meta_df_5_core["image_title_based_desc"][id]

    for id, row in meta_df_5_core.iterrows():
        for col in ["title", "brand"]:
            if pd.isna(row[col]):
                meta_df_5_core.at[id, col] = "" 


    meta_df_5_core["text_concat"] = args.dataset + " " + ("" if text_column == "title" else meta_df_5_core["title"]) + " " + \
                                meta_df_5_core["description"] + " " + \
                                meta_df_5_core["brand"] + meta_df_5_core[text_column]

    return meta_df_5_core

def get_text_features(args, model, df, text_column):
    """
    Get text features for the specified text column in the DataFrame.
    """
    txt_embeddings = model.encode(df["text_concat"], show_progress_bar=True, normalize_embeddings=True)
    np.save(os.path.join(f"/data/{args.dataset}/{text_column}_txt_feat.npy"), txt_embeddings)

def main():
    args = arg_parse()

    if not os.path.exists("data"):
        raise FileNotFoundError("Could not find the data directory. Please ensure it exists.")

    if not os.path.exists(os.path.join("data", args.dataset)):
        raise FileNotFoundError(f"Could not find the dataset directory for {args.dataset}. Please ensure it exists.")
    
    model = get_encoding_model(args.txt_embedding_model)
    dataset_folder = get_dataset(args)
    detail_dataset_df = get_details_dataset_df(args, dataset_folder["mapping_file"], 
                                               dataset_folder["five_core_data"], 
                                               dataset_folder["description_file"], 
                                               args.text_column)
    meta_df_5_core_after_concat = concat_with_meta_data(args, 
                                                        detail_dataset_df,
                                                        dataset_folder["meta_data"], 
                                                        args.text_column)

    get_text_features(args, 
                      model=model,
                      df=meta_df_5_core_after_concat,
                      text_column=args.text_column)

if __name__ == "__main__":
  main()

