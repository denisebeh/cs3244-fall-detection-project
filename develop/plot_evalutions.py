from matplotlib import pyplot as plt
import json

with open('./plots/results.json', "r") as f:
    results = json.load(f)

sensitivity = []
specificity = []
far = []
mdr = []
accuracy = []
x = []

for key in results.keys():
    x.append(int(key))
    sensitivity.append(results[key]["sensitivity_mean"])
    specificity.append(results[key]["specificity_mean"])
    far.append(results[key]["far_mean"])
    mdr.append(results[key]["mdr_mean"])
    accuracy.append(results[key]["accuracy_mean"])

# plot sensitivity vs epoch
fig = plt.figure()
plt.plot(x, sensitivity)
plt.title('Sensitivity vs Epoch')
plt.ylabel("Sensitivity")
plt.xlabel("Epoch")
plt.savefig("./plots/analysis/sensitivity.png")
plt.close(fig)

# plot specificity vs epoch
fig = plt.figure()
plt.plot(x, specificity)
plt.title('Specificity vs Epoch')
plt.ylabel("Specificity")
plt.xlabel("Epoch")
plt.savefig("./plots/analysis/specificity.png")
plt.close(fig)

# plot FAR vs epoch
fig = plt.figure()
plt.plot(x, far)
plt.title('FAR vs Epoch')
plt.ylabel("FAR")
plt.xlabel("Epoch")
plt.savefig("./plots/analysis/far.png")
plt.close(fig)

# plot MDR vs epoch
fig = plt.figure()
plt.plot(x, mdr)
plt.title('MDR vs Epoch')
plt.ylabel("MDR")
plt.xlabel("Epoch")
plt.savefig("./plots/analysis/mdr.png")
plt.close(fig)


# plot accuracy vs epoch
fig = plt.figure()
plt.plot(x, accuracy)
plt.title('Accuracy vs Epoch')
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.savefig("./plots/analysis/accuracy.png")
plt.close(fig)