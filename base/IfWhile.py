# 条件判断  elif是else if的缩写，完全可以有多个elif，所以if语句的完整形式就是
# 因为input()返回的数据类型是str，str不能直接和整数比较，必须先把str转换成整数。Python提供了int()函数来完成这件事情：
age = int(input("请输入年龄:"))
if 10 > age > 5:
    print("哈哈哈")
elif 20 > age >= 10:
    print("嘿嘿")
else:
    print("哈哈哈")


