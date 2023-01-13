pos_num = 1000
neg_num = 750
def f(percent):
    pos_weight = round(pos_num * percent)
    return pos_weight
    neg_weight = round(neg_num * percent)
    return neg_weight
pos_weight, neg_weight = f(0.5)
print(f(0.5))
print(pos_weight)
print(neg_weight)





'''
pos_num = 1000
neg_num = 750
def pos_amount(percent):
    pos_weight = round(pos_num * percent)

def neg_weight(percent):
    neg_weight = round(neg_num * percent)

print(amount(0.05))
print(pos_amount)
'''