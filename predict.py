from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import time, pprint, datetime 
import torch

def log():
    '''warning 時輸出 log'''
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    print(f"是否使用 GPU: {torch.cuda.is_available()}")

def predict():
    '''計時開始'''
    tStart = time.time() 

    '''測試資料'''
    test_csv = pd.read_csv('test.csv')

    '''放置符合測試格式的資料'''
    test_data = []

    '''資料轉換'''
    list_dataset = test_csv.values.tolist()
    for dataset in list_dataset:
        test_data.append(dataset[1])
    
    '''pre-trained model、batch size 與 epoch'''
    model = 'deberta'
    model_name_prefix = 'microsoft/'
    model_name_main = 'deberta-base'
    model_name = model_name_prefix + model_name_main
    batch_size = 156
    epoch = 10

    '''output 資料夾'''
    output_dir = f"outputs/{model_name_main}-bs-{batch_size}-ep-{epoch}-cls-model/"
    # checkpoint-41355-epoch-15/
    # checkpoint-27570-epoch-10/
    # checkpoint-19299-epoch-7/

    '''自訂參數'''
    model_args = ClassificationArgs()
    model_args.train_batch_size = batch_size
    model_args.num_train_epochs = epoch
    model_args.overwrite_output_dir = False
    model_args.reprocess_input_data = False
    model_args.use_multiprocessing = False
    model_args.save_model_every_epoch = False
    model_args.save_steps = -1
    # model_args.learning_rate = 4e-5
    model_args.output_dir = output_dir

    '''迴歸分析才需要設定'''
    # model_args.num_labels = 1
    # model_args.regression = True

    '''建立 ClassificationModel'''
    model = ClassificationModel(model, output_dir, use_cuda=torch.cuda.is_available(), cuda_device=0, args=model_args)

    '''預測結果'''
    predictions, raw_outputs = model.predict(test_data)

    '''放置整合預測結果'''
    list_result = []

    '''整合預測結果'''
    for index, sentiment in enumerate(predictions):
        list_result.append([
            list_dataset[index][0],
            sentiment
        ])

    '''將結果存回 excel'''
    # strDataTime = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
    result_df = pd.DataFrame(list_result)
    result_df.columns = ["ID", "sentiment"]
    result_df.to_csv(f"{model_name_main}-bs-{batch_size}-ep-{epoch}.csv", index=False)

    '''計時結束'''
    tEnd = time.time()

    '''輸出程式執行的時間'''
    print("It cost %f sec" % (tEnd - tStart))

if __name__ == "__main__":
    log()
    predict()
