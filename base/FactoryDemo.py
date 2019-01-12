# 工厂模式


class Person:

    def __init__(self, name):
        self.name = name;

    # def __init__(self, name, age):
    #     self.name = name
    #     self.age = age

    def work(self):
        print( self.name + "开始工作了")
        axe = StoneAxe();
        axe.cut_tree();

class Axe(object):
    def cut_tree(self):
        print("正在砍树");

class StoneAxe(Axe):
    def cut_tree(self):
        print("使用石头做的斧头砍树");

class SteeAxe(Axe):
    def cut_tree(self):
        print("使用钢斧头砍树");

p = Person('gfadf 男');
p.work();