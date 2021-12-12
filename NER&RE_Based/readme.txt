
1.对原始数据ori_data进行预处理
python BuildData.py

2.训练词向量
python Pretrain_Word_Embedding.py

3.
tplinker：对实体和关系做联合提取
可得到如下tag:
(device_type, vendor, null)
(device_type, null, product)
(null, vendor, product)
(device_type, vendor, product)

tplinker_plus：对实体做提取
可得到如下tag:
(device_type, null, null)
(null, vendor, null)
(null, null, product)

训练
python train.py

在测试集上进行评估，将预测结果保存至results
python evalution.py

将预测结果和测试集数据转换成devtag形式，存至results
python res2tag.py

对转换形式后的结果做评估，存至reports
python metrics.py

4.上传数据文件, 通过-T选择要标注的tag:<dvp, pd, vd, vp, d, v, p>
for example:
python run.py -f ./test/test-dvp.json -T dvp