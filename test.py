from string import punctuation

import re

str = "aksjnekljfklen正"

temp = re.sub('[a-zA-Z]', '', str)
print(temp)

str = "《三国演义》中的“水镜先生”是司马徽56585622"
add_punc='0123456789-' # 自定义--数字
all_punc = punctuation + add_punc
print(punctuation)
print('s' in all_punc)