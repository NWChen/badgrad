from abc import ABC, abstractmethod

class Operator(ABC):

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def grad(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Add(Operator):

    def __init__(self, l: Operator, r: Operator):
        self.l = l
        self.r = r
    
    def val(self):
        return self.l.val() + self.r.val()
    
    def grad(self):
        return Add(self.l.grad(), self.r.grad())

    def __str__(self):
        return f'({self.l} + {self.r})'


class Sub(Operator):

    def __init__(self, l: Operator, r: Operator):
        self.l = l
        self.r = r

    def val(self):
        return self.l.val() - self.r.val()
    
    def grad(self):
        return Sub(self.l.grad(), self.r.grad())
    
    def __str__(self):
        return f'({self.l} - {self.r})'


class Mult(Operator):

    def __init__(self, l: Operator, r: Operator):
        self.l = l
        self.r = r

    def val(self):
        return self.l.val() * self.r.val()
    
    def grad(self):
        return Add(
            Mult(self.left, self.right.grad()),
            Mult(self.left.grad(), self.right),
        )
    
    def __str__(self):
        return f'({self.l} * {self.r})'


class Constant(Operator):
    
    def __init__(self, k):
        self.k = k
    
    def val(self):
        return self.k
    
    def grad(self):
        return Constant(0)
    
    def __str__(self):
        return f'{self.k}'


class Pow(Operator):

    def __init__(self, base: Operator, exp: Operator):
        self.base = base
        self.exp = exp
    
    def val(self):
        return self.base.val() ** self.exp.val()
    
    def grad(self):
        # y = g(x)^n 
        # dy/dx = n(g(x)^(n-1))(dg/dx)
        return Mult(
            Mult(
                self.exp,
                Pow(
                    self.base,
                    Add(self.exp, Constant(-1))
                )
            ),
            self.base.grad()
        )

    def __str__(self):
        return f'({self.base}^{self.exp})'


class X(Operator):

    def __init__(self):
        pass

    def val(self, x):
        return x
    
    def grad(self):
        return Constant(1)
    
    def __str__(self):
        return 'x'

if __name__ == '__main__':

    # f(x) = (x - 1)^2 + x
    expr = Add(Pow(Sub(X(), Constant(1)), Constant(2)), X())
    print(expr)
    print(expr.grad())
