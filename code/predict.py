import os

from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存，按需分配

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from keras.models import Model, load_model, Sequential
from keras.regularizers import l1, l2
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.utils import to_categorical
import joblib


def one_hot(data, windows=50):
    # define input string
    data = data
    length = len(data)
    print("length:",length)
    # define empty array

    # data_X = np.zeros((length, 2*windows+1, 4))
    data_X = np.zeros((length, windows, 5))
    data_Y = []
    for i in range(length):
        x = data[i].split() #通过制定分隔符对字符串进行切
        data_Y.append(int(float(x[2])))
        nucl = 'ACGUNDEFHIKLMPQRSVWY-BJOTXZ'
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(nucl))
        integer_encoded = [char_to_int[char] for char in x[1]]
        j = 0
        for value in integer_encoded:
            if value in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25, 26]:
                # for k in range(5):
                for k in range(5):
                    data_X[i][j][k] = 0.05
            else:
                data_X[i][j][value] = 1.0
            j = j + 1
    data_Y = np.array(data_Y)
    return data_X, data_Y


def perform_eval_1(predictions, Y_test, verbose=0):
    # class_label = np.uint8([round(x) for x in predictions[:, 0]]) # round()函数进行四舍五入
    # R_ = np.uint8(Y_test)
    # R = np.asarray(R_)
    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))

    CM = metrics.confusion_matrix(R, class_label, labels=None)
    CM = np.double(CM)  # CM[0][0]：TN，CM[0][1]：FP，CM[1][0]：FN，CM[1][1]：TP

    # 计算各项指标
    sn = (CM[1][1]) / (CM[1][1] + CM[1][0])  # TP/(TP+FN)
    sp = (CM[0][0]) / (CM[0][0] + CM[0][1])  # TN/(TN+FP)
    acc = (CM[1][1] + CM[0][0]) / (CM[1][1] + CM[0][0] + CM[0][1] + CM[1][0])  # (TP+TN)/(TP+TN+FP+FN)
    pre = (CM[1][1]) / (CM[1][1] + CM[0][1])  # TP/(TP+FP)
    f1 = (2 * CM[1][1]) / (2 * CM[1][1] + CM[0][1] + CM[1][0])  # 2*TP/(2*TP+FP+FN)
    mcc = (CM[1][1] * CM[0][0] - CM[0][1] * CM[1][0]) / np.sqrt(
        (CM[1][1] + CM[0][1]) * (CM[1][1] + CM[1][0]) * (CM[0][0] + CM[0][1]) * (
                    CM[0][0] + CM[1][0]))  # (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^1/2
    gmean = np.sqrt(sn * sp)
    auroc = metrics.roc_auc_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")
    aupr = metrics.average_precision_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")

    if verbose == 1:
        print("Sn(Recall):", "{:.4f}".format(sn), "Sp:", "{:.4f}".format(sp), "Acc:", "{:.4f}".format(acc),
              "Pre(PPV):", "{:.4f}".format(pre), "F1:", "{:.4f}".format(f1), "MCC:", "{:.4f}".format(mcc),
              "G-mean:", "{:.4f}".format(gmean), "AUROC:", "{:.4f}".format(auroc), "AUPR:", "{:.4f}".format(aupr))

    # return [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr]
    return sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr


# 说明： 实验结果保存到文件
# 输入： 文件标识符和结果
# 输出： 无
def write_res_1(filehandle, res, fold=0):
    filehandle.write("Fold: " + str(fold) + " ")
    filehandle.write("Sn(Recall): %s Sp: %s Acc: %s Pre(PPV): %s F1: %s MCC: %s G-mean: %s AUROC: %s AUPR: %s\n" %
                     ("{:.4f}".format(res[0]),
                      "{:.4f}".format(res[1]),
                      "{:.4f}".format(res[2]),
                      "{:.4f}".format(res[3]),
                      "{:.4f}".format(res[4]),
                      "{:.4f}".format(res[5]),
                      "{:.4f}".format(res[6]),
                      "{:.4f}".format(res[7]),
                      "{:.4f}".format(res[8]))
                     )
    filehandle.flush()
    return


def plot_metric(history, metric, fold_count=0):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)

    with plt.style.context(['ggplot', 'grid', 'no-latex']):
        plt.plot(epochs, train_metrics, 'b--')
        plt.plot(epochs, val_metrics, 'r-')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        # if fold_count == 0:
        #     plt.savefig('./images/' + metric + '.jpg', dpi=600, bbox_inches='tight')
        # else:
        #     plt.savefig('./images/' + metric + '_' + str(fold_count) + '_fold.jpg', dpi=600, bbox_inches='tight')
        # plt.show()
        plt.savefig('F:/py-file/weizhangying/DNA/DNA_squence/m6Am/result_images1/' + metric + '_' + str(
            fold_count + 1) + '_fold_three_stacking.jpg', dpi=600, bbox_inches='tight')
        plt.show()


def acc_loss_plot(train_loss, train_acc, val_loss, val_acc, fold_count=0):
    with plt.style.context(['ggplot', 'grid', 'no-latex']):
        # 创建一个图
        plt.figure()
        plt.plot(train_loss, label='train loss')
        plt.plot(train_acc, label='train acc')
        plt.plot(val_loss, label='val loss')
        plt.plot(val_acc, label='val acc')
        # plt.grid(True, linestyle='--', alpha=0.5)  # 增加网格显示
        plt.title('acc-loss')  # 标题
        plt.xlabel('epoch')  # 给x轴加注释
        plt.ylabel('acc-loss')  # 给y轴加注释
        plt.autoscale(tight=True)  # 自动缩放(紧密)
        plt.legend(loc="upper right")  # 设置图例显示位置
        # if fold_count == 0:
        #     plt.savefig('images/acc-loss.jpg', dpi=600, bbox_inches='tight')  # bbox_inches可完整显示
        # else:
        #     plt.savefig('images/acc-loss_' + str(fold_count) + '_fold.jpg', dpi=600, bbox_inches='tight')
        # plt.show()
        plt.savefig(
            'F:/py-file/weizhangying/DNA/DNA_squence/m6Am/result_images1/acc-loss_' + str(fold_count + 1) + '_fold__three_stacking.jpg',
            dpi=600, bbox_inches='tight')
        plt.show()

# return [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr]
def performance_mean(performance):
    # print('Sn = %.2f%% ± %.2f%%' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    # print('Sp = %.2f%% ± %.2f%%' % (np.mean(performance[:,  1]), np.std(performance[:, 1])))
    # print('Acc = %.2f%% ± %.2f%%' % (np.mean(performance[:,  2]), np.std(performance[:, 2])))
    # print('Pre = %.2f%% ± %.2f%%' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    # print('F1 = %.2f%% ± %.2f%%' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    # print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 5]), np.std(performance[:, 5])))
    # print('Gmean = %.4f ± %.4f' % (np.mean(performance[:, 6]), np.std(performance[:, 6])))
    # print('Auroc = %.4f ± %.4f' % (np.mean(performance[:,  7]), np.std(performance[:, 7])))
    # print('Aupr = %.4f ± %.4f' % (np.mean(performance[:, 8]), np.std(performance[:, 8])))
    print('Sn = %.4f ± %.2f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.2f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f± %.2f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    # print('Pre = %.2f%% ± %.2f%%' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    # print('F1 = %.2f%% ± %.2f%%' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    # print('Gmean = %.4f ± %.4f' % (np.mean(performance[:, 6]), np.std(performance[:, 6])))
    print('Auroc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('Aupr = %.4f ± %.4f' % (np.mean(performance[:, 5]), np.std(performance[:, 5])))

# 性能评价指标
def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    # 计算敏感性Sn
    Sn = TP / (TP + FN + 1e-06)
    # 计算特异性Sp
    Sp = TN / (FP + TN + 1e-06)
    # 计算Acc值
    Acc = (TP + TN) / len(y_true)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC


def model_pred(test, test_label):
    train_pred_D = np.zeros(len(test))
    for i in range(10):     # DenseNet
        model_D = load_model('D_model/stacking_model_{}_fold.h5'.format(i + 1))
        train_pred_D += model_D.predict(test)[:, 1] / 10  # 对测试集的预测，十折交叉验证得十个预测值取平均

    train_pred_A = np.zeros(len(test))
    for i in range(10):     # ANN
        model_A = load_model('A_model/stacking_model_{}_fold.h5'.format(i + 1))
        train_pred_A += model_A.predict(test)[:, 1] / 10  # 对测试集的预测，十折交叉验证得十个预测值取平均

    train_pred_C = np.zeros(len(test))
    for i in range(10):     # CreateModel
        model_C = load_model('C_model/stacking_model_{}_fold.h5'.format(i + 1))
        train_pred_C += model_C.predict(test)[:, 1] / 10  # 对测试集的预测，十折交叉验证得十个预测值取平均

    train_pred_C = pd.DataFrame(train_pred_C)
    train_pred_D = pd.DataFrame(train_pred_D)
    train_pred_A = pd.DataFrame(train_pred_A)
    # 将三个模型预测值进行拼接后便于与train进行拼接
    train_stack = pd.concat([train_pred_C, train_pred_A, train_pred_D], axis=1)

    sequence_feature = np.reshape(test, (len(test), 41 * 5))
    sequence_feature = pd.DataFrame(sequence_feature)
    train_stack = pd.DataFrame(train_stack)
    sequence_feature = pd.concat([sequence_feature, train_stack], axis=1)

    # 用于存放测试集的预测
    sequences_pred = np.zeros(len(sequence_feature))

    for fold_ in range(10):                                                 #(0.797-->>0.787324)
        # model = joblib.load('model3/stacking_model_update_two_{}_fold.h5'.format(fold_ + 1))  # 0.798592  (0.8098)(0.8102)
        model = joblib.load('model2/stacking_model_update_two_{}_fold.h5'.format(fold_ + 1))  # 0.805634
        # model = joblib.load('model1/stacking_model_update_two_{}_fold.h5'.format(fold_ + 1))  # Acc = 0.804225 (0.81)
        # model = joblib.load('model4/stacking_model_update_two_{}_fold.h5'.format(fold_ + 1))  # Acc = 0.795775  (0.8084)
        # model = joblib.load('model5/stacking_model_update_two_{}_fold.h5'.format(fold_ + 1))  # Acc = 0.795775  (0.8102)



        # Step 5: 得到预测值
        sequences_pred += model.predict(sequence_feature) / 10  # 对测试集的预测，十折交叉验证得十个预测值取平均

    #######################################################
    y_pred = sequences_pred  # 预测概率值

    Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], y_pred)
    AUC = roc_auc_score(test_label[:, 1], y_pred)
    Aupr = average_precision_score(test_label[:, 1], y_pred)

    # performance[fold_, :] = np.array((Sn, Sp, Acc, MCC, AUC, Aupr))
    print('模型独立测试的结果： Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, Aupr = %f' % (Sn, Sp, Acc, MCC, AUC, Aupr))




if __name__ == '__main__':
    #
    # 超参数设置
    BATCH_SIZE = 64  # 批次大小一般设为：16、32、64、128、256,大 batch size 占用空间，小 batch size 消耗时间
    N_EPOCH = 25
    WINDOWS = 41  # 从文件读取序列片段（训练+验证，阳性+阴性）
    f_r_train = open("m6Am_dataset/df_train_data.txt", "r", encoding='utf-8')
    f_r_test = open("m6Am_dataset/df_test_data.txt", "r", encoding='utf-8')

    # 训练序列片段构建
    train_data = f_r_train.readlines()

    # 预测序列片段构建
    test_data = f_r_test.readlines()

    # 关闭文件
    f_r_train.close()
    f_r_test.close()


    # one_hot编码序列片段
    train, train_Y = one_hot(train_data, windows=WINDOWS)
    train_Y = to_categorical(train_Y, num_classes=2)
    test, test_Y = one_hot(test_data, windows=WINDOWS)
    test_Y = to_categorical(test_Y, num_classes=2)

    print("train的预测结果：")
    model_pred(train, train_Y)
    print("test的预测结果：")
    model_pred(test, test_Y)


