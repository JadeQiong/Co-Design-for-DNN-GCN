import matplotlib.pyplot as plt


cnt = 0
test = []
train = []
p_test = []
p_train = []

test_loss = []
train_loss = []
p_test_loss = []
p_train_loss = []

with open('C:/Users/87179/Desktop/GCN_cora.txt', "r") as f:
    for line in f:
        cnt += 1
        if cnt <= 3 or cnt > 103:
            continue
        txts = line.split(' ')
        a, c = txts[2].split('=')
        a, b = txts[4].split('=')
        a, d = txts[1].split('=')
        a, e = txts[3].split('=')
        test.append(float(b))
        train.append(float(c))
        train_loss.append(float(d))
        test_loss.append(float(e))


cnt = 0
with open('C:/Users/87179/Desktop/GCN_cora_p.txt', "r") as ff:
    for line in ff:
        cnt += 1
        if cnt <= 3 or cnt > 103:
            continue
        txts = line.split(' ')
        a, c = txts[2].split('=')
        a, b = txts[4].split('=')
        a, d = txts[1].split('=')
        a, e = txts[3].split('=')
        print(line)
        p_test.append(float(b))
        p_train.append(float(c))
        p_train_loss.append(float(d))
        p_test_loss.append(float(e))

x_axis_data = [i for i in range(1, 51)]
print(len(test))
print(train)
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.plot(x_axis_data, train[:50], label="train")
plt.plot(x_axis_data, test[:50], label="validation", color = 'r')
plt.legend()
# plt.plot(x_axis_data, train_loss)
# plt.plot(x_axis_data, test_loss)
plt.savefig("cora_without_p.png")
plt.close()

plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.plot(x_axis_data, p_train[:50], label="train with pruning")
plt.plot(x_axis_data, p_test[:50], label="validation with pruning", color = 'r')
plt.legend()
plt.savefig("cora_p.png")
