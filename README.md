# Name Ethnicity Classifier

Humans are often able to accurately predict someone's ethnicity based on their last name, even if they have not seen it before. For example, one can predict that "Kang" is more likely to be a Chinese last name than a Japanese one, while "Kobayashi" seems characteristically Japanese. 

I used a recurrent neural network with TensorFlow in Python to train a model to do this automatically. The model achieved **97% accuracy** in classifying names as Chinese pr Japanese names, 87% accuracy in classifying names as Chinese, Japanese, or Vietnamese, and 79% accuracy in classifying names as Chinese, Japanese, Vietnamese, or Korean. 

This drastic decrease in accuracy with the introduction of Vietnamese and Korean makes sense, and would likely also be seen among humans, as Korean and Vietnamese names seem similar to Chinese names. Even within the dataset, some Korean and Vietnamese names were the same as Chinese names ("Tien" is listed both as Chinese and Vietnamese, while "Wang" is listed both as Chinese and Korean).

Name data was scraped from [familyeducation.com](https://www.familyeducation.com/baby-names/browse-origin/surname).