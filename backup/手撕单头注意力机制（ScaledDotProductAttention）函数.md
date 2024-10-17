# 手撕单头注意力机制（ScaledDotProductAttention）函数
输入是query和 key-value，注意力机制首先计算query与每个key的关联性（compatibility），每个关联性作为每个value的权重（weight），各个权重与value的乘积相加得到输出。
![image](https://github.com/user-attachments/assets/70d11f21-5f1e-4990-8422-28f21b7fa25b)
1. 为什么要缩放：
如果不对softmax的输入做缩放，那么万一输入的数量级很大，softmax的梯度就会趋向于0，导致梯度消失。
2. attention mask的机制
attention mask 有效的标记为1，无效的标记为0，之后与attention score矩阵相加 1的地方不变 0的地方会加一个非常大的负数 从而利用softmax(负数)无限趋近于0的特点，避免padding的token的影响
3. k.transpose(1, 2)或者torch.transpose(a, 1, 0)不是原地操作
4. transpose() 和 permute()
 1.都是返回转置后矩阵。
 2.都可以操作高纬矩阵，permute在高维的功能性更强。
```
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # bmm：batch matrix matrix product https://blog.csdn.net/qq_39407949/article/details/132890694
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul 
        u = u / self.scale # 2.Scale


        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64

    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)
    mask = torch.zeros(batch, n_q, n_k).bool()

    attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
    attn, output = attention(q, k, v, mask=mask)

    print(attn)
    print(output)
```