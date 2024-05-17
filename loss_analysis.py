import pandas as pd
import matplotlib.pyplot as plt
dict_loss={
            "loss": [],
            "prediction_loss":[],
            "score_loss":[],
            "plan_loss":[],
            "plan_cost":[]
        }
# print("dict_loss.keys:",list(dict_loss.keys()))
df=pd.read_csv("/home/oem/zkf/planTF-dipp/loss_train_csv.csv"
,header=None,
 names=list(dict_loss.keys()))
df = df.dropna(how='all')
print(df.head)


plt.plot(df['plan_cost'][5800:], label='plan_cost')
# plt.plot(df['plan_loss'], label='plan_loss')
# plt.plot(df['score_loss'], label='score_loss')
# plt.plot(df['prediction_loss'], label='prediction_loss')
# plt.plot(df['loss'][:5000], label='loss')
# plt.plot(df['for_al_loss_heading'], label='For AL Loss Heading')
# plt.plot(df['bc_loss'][2000:4000], label='BC Loss')
# plt.plot(df['agent_reg_loss'][2000:4000], label='Agent Reg Loss')
# plt.plot(df['safe_loss'][4000:8000], label='Safe Loss')
# plt.plot(df['for_al_loss_acc'][4000:8000], label='For AL Loss Acc')
# plt.plot(df['for_al_loss_jerk'][4000:8000], label='For AL Loss Jerk')
# plt.plot(df['for_al_loss_heading'][4000:8000], label='For AL Loss Heading')
# plt.xlabel('BATCH')
plt.ylabel('Loss Value')
plt.legend()
plt.show()