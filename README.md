# Detecting fraudulent Bitcoin transactions using supervised machine learning methods


## 1. Introduction
### 1.1 Problem Definition

Since the inception of Bitcoin in 2009, there have been concerns with regards to its security and the potential for it to be used in illegal activity, partly due to its pseudo-anonymous properties. Transactions over the Bitcoin network are not directly linked to personal identities, although this can be established through deeper investigation as more often than not, the Bitcoins are purchased using personal bank accounts on centralised exchanges, which often require proof of identity. 

Nevertheless, Bitcoin fraud is still an inevitability. For those who know how to hide their identity, Bitcoin provides a better alternative to traditional currencies for malicious purposes. Dark web marketplaces such as the infamous Silk Road heavily relied on the use of cryptocurrencies such as Bitcoin in order to facilitate the transfer of illegal weapons and drugs amongst users.

Over the last decade, there have been multiple high profile cases of centralised exchanges being hacked, resulting in the leaking of thousands of Bitcoin private keys - which give a user access to their Bitcoin funds. Once leaked, hackers can use these private keys to transfer funds from these addresses to another address, one which only the hackers have the private key for. Since Bitcoin transactions are non-reversible, there is no way for customers to retrieve their funds.
The aim of this project is to develop a machine learning model which can identify fraudulent Bitcoin transactions.

### 1.2 Machine Learning Methods

#### 1.2.1 Anomaly Detection
Anomaly detection (AD) is a broad technique which aims to identify data points which deviate from the majoriy of the data. AD can be used in the identification of rare events or observations, where features of these events are significantly different from normal instances. Thus, AD has much practical importance due to it's broad applications in defense against cyber-crimes, fraudulent activity and much more. Machine learning models that aim to detect anomalous data points can be of three main types: unsupervised, semi-supervised and supervised. The correct method depends on the availability of labels in the dataset.

Rather than anomaly detection models learning what makes a certain data point abnormal, AD models learn what the 'normal' data points consists of. Consequently, any data points that fall outside of that defined normal, are marked as outliers, or anomalous.

#### 1.2.2 Fraud Detection
Fraud detection focuses specifically on identiying fraudulent data points or activities, and is particularly applicable for detecting fraudulent network activity or fraudulent transactions as in the case of credit card fraud. Fraud detection can be treated as anomaly detection or a classification problem, again, depending on the characteristics of the dataset and the goal of the machine learning model. This is particularly useful in cases where fraud patterns are evolving and may not be well-defined.

#### 1.2.3 Binary Classification
Binary classification is a machine learning technique where the goal is to categorise data points into one of two classes, or categories. It is a broader concept that can include both anomaly detection and fraud detection, but is not limited to these speciic use cases. Binary classification can only be performed when the data has been clearly labled, with data belonging to either class.

### 1.3 Assumptions and goals
The goal of this project is to explore the various supervised learning methods to tackle the problem of fraud detecction for Bitcoin transactions. Supervised learning models require the data to be labelled and due to the fact that fraudulent transactions are far more rare when compared with legitimate transactions, it is extermely likely that avaiable datasets will be heavilty skewed in favour of legit transactions, where instances of known fraudulent transactions will be very limited in number. This will pose a significant challenge and so steps will need to be taken to address this challenge.

There are two ways I can approach this problem:
1. Treat fraud detection as a binary classification problem:
   - Train a binary classifier such as MLP or another form of ANN to predict transactions as either fraudulent or legit using a labelled dataset of known legit and fraud transactions.
   - I can use some other supervised learning model that has been used in literature such as SVM or a simple logistical regression model to compare my neural network to.

2. Treat fraud detection as an anomaly detection problem:
   - Using some supervised learning method, train a model to learn what is a 'normal' or legitimate transaction and then feed data into this model. Fradulent transactions should then be identofied as they will differ from the legit transactions in some metric.
   - Can use an autoencoder for this
   - There has been research into Graph Neural Networks (GNN) so could explore that
   - Other neural network architetures could potentially be used such as RNN
   - I can then compare the performance of these models to some other out-of-the-box anomaly detection model such as dbscan clustering.

## 2. Background

### 2.1 Bitcoin

Bitcoin is a decentralised, peer-to-peer money ecosystem which allows participants to send and receive value using nothing more than a computer and an internet connection. In the Bitcoin network, transactions are made directly between participants, without the need for a middle-man. Bitcoin transactions are final and cannot be reversed. It is therefore extremely important that users are always in control of the bitcoins they own. This ownership is proven through special ‘keys’ which allow the owners to ‘unlock’ bitcoin funds which have been sent to them. Therefore, whoever controls the keys, controls the bitcoins. 

**2.1.1 Transaction Inputs and outputs**

![Overview of Bitcoin Transaction Inputs and Outputs](https://en.bitcoin.it/w/images/en/f/f1/Bitcointransactions.JPG)

In a bitcoin transaction, there is nothing being ‘sent’ per say. Rather, the owner of a certain number of bitcoins specifies who the new owner will be. They first ‘lock’ these funds using a special locking script and a unique digital signature. These funds can now only be unlocked, or spent, by the user that owns a particular key, a key that the old owner specifies. This forms the _output _of a transaction. The outputs of a previous transaction become part of the input to the next transaction when a new owner wants to spend the funds that were sent to them by unlocking the unspent transaction outputs (UTXO) using their special key.

### 2.2 Fraudulent Transactions

**What is an fraudulent transaction?**

An anomalous transaction is one that contains data points which are significantly different from those of a normal transaction. This can come in many forms. This paper describes some of the typical types or categories of anomalous transactions:

**2.2.1 Bitcoin theft**

This usually occurs when a malicious actor gains access to a victim’s private key, giving them complete control over the Bitcoin funds in the victim’s wallet. This can be a result of a breach or hack of a centralised cryptocurrency exchange in which the user's information is leaked; or it can be a result of a scam where victims are tricked into giving scammers access to their wallets voluntarily. Regardless of the method of acquisition, these types of thefts have some common patterns.

* Victim’s wallet is quickly emptied, with all funds being transferred to the thief’s wallet in a **single transaction output**.
* Fraudulent transactions may involve **abnormal fees in relation to the transaction amount. **Miners will prioritise transactions with large fees and so those trying to transfer stolen funds as quickly as possible may introduce large fees for these transactions.

**2.2.2 Laundering or other malicious activities**

As mentioned, Bitcoin is a better digital currency alternative than traditional currencies to use for illegal activities because of the pseudo-anonymity it provides, as well as its peer-to-peer nature. Thus, it is used by criminal, drug cartels and crime syndicates for cross-border transactions for purposes of laundering illegal money. It is worth noting that there are far better cryptocurrencies for this purpose such as privacy coins like Monero, where transactions are completely untraceable and anonymous; however Bitcoin is still used for illegal activities due to its popularity in relation to other cryptocurrencies. 

Pattern of money laundering may include:

* Tainting of bitcoins whereby some possibly **stolen funds are mixed with other, legal funds** and used together as inputs to a new transaction, thus convoluting the transaction making it more difficult to detect.
* Thief's may try to mask their actions by making **numerous, smaller transaction outputs** **in a short space of time** in an attempt to confuse the victim and authorities
* Making transactions at **unusual times of the day** to avoid detection.

**2.2.3 Crypto Ransom & Extortion**

**Bitcoin is also a popular currency for **perpetrators of ransoms and exploitation. A characteristic of these types of crimes are that criminals will often force victims to make a** single or a few large transactions to a single address** or potentially make smaller, **more numerous transactions to different addresses**, again in order to confuse investigators after the fact.

**2.2.4 Network Attacks**

Cybercriminals may try to attack the Bitcoin network in an attempt to disrupt the normal functioning or exploit vulnerabilities in transaction processing, by making malicious transactions. Patterns may include:

* _Transaction spamming_ or _DDOS_ attack where cybercriminals flood the network with **numerous small transactions with very low fees**. This creates congestion and a build up of many unconfirmed transactions as miners stay clear of these low fee transactions in favour of more profitable transactions.
* _Blockchain bloat_: Attackers may attempt to bloat the size of the blockchain by creating a **large number of large transactions** in an attempt to increase the storage requirements, leading to slow synchronisation across the network.

From these behaviours, some typical transaction markers should be explored and used to train the machine learning model. This small subset of characteristics can help to detect malicious behaviours:

<table>
  <tr>
   <td>Type of Anomaly
   </td>
   <td>Transactional markers
   </td>
  </tr>
  <tr>
   <td><strong>Theft</strong>
   </td>
   <td>
<ul>

<li>Single (or few) transaction outputs

<li>Higher fees
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>Laundering
   </td>
   <td>
<ul>

<li>Poisoning (funds from illegal activities mixed with funds from legal activities)

<li>Numerous small transactions to various addresses

<li>Late hours of the day
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>Ransom
   </td>
   <td>
<ul>

<li>Single, large outputs to a single address

<li>numerous , smaller outputs to multiple addresses (similar to laundering)
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>Network Attacks
   </td>
   <td>
<ul>

<li>Numerous small transactions with very low fees

<li>numerous , large transactions with low fees
</li>
</ul>
   </td>
  </tr>
</table>

Therefore, the machine learning model should be able to learn from these transaction characteristics to detect anomalous transactions:

* Numerous, small transactions to multiple addresses (laundering/network attacks)
* Few, large transactions to a single address (theft/ransoms)
* Many, large transactions to many addresses (network attacks)
* Very low fees (network attacks)
* Very High fees (theft or ransoms)

While these can be indicative of suspicious activities, it's important to note that these characteristics are not definitive proof of malicious intent, and legitimate transactions might exhibit similar patterns. Nevertheless, these markers can be useful for flagging transactions for further investigation and will be the focus of this project.

### 2.3 Problems with this description

An issue with categorising malicious activity using transaction markers is that there will undoubtedly be scenarios where the characteristics described may be associated with normal transaction behaviour:

**2.3.1 Numerous Small Transactions:**

* Microtransactions: Some businesses or applications rely on microtransactions, where small amounts of cryptocurrency are transferred frequently. This is common in gaming or content platforms.
* Dollar-Cost Averaging: Investors might engage in dollar-cost averaging, making frequent small purchases of cryptocurrency over time.

**2.3.2 Higher Fees:**

* Urgency or Priority Transactions: Users might intentionally choose higher fees to ensure quicker confirmation times for their transactions, especially during periods of network congestion.
* Complex Smart Contract Execution: Certain transactions involving complex smart contracts or advanced features may require higher fees.

**2.3.3 Large Outputs to Single Address:**

* Exchange Withdrawals: When users withdraw funds from a cryptocurrency exchange, a large output to a single address is common.
* Business Transactions: Large payments between businesses or entities can result in a single, large output.

**2.3.4 Numerous Small Outputs to Multiple Addresses:**

* Payment Splitting: In a business context, payments might be split into numerous smaller outputs to distribute funds among multiple recipients.
* Wallet Shuffling: Some users intentionally split their funds into smaller amounts and distribute them across multiple addresses for privacy reasons.

**2.3.5 Late Hours of the Day:**

* Global Nature of Cryptocurrency Markets: Participants in cryptocurrency markets are distributed globally, leading to transactions occurring at various times, including late hours.

Regardless, the transaction markers described can still provide a useful indication of potential suspicious or malicious activity.


## 3. Dataset Description

This project uses the [Bitcoin Network Transactional Metadata dataset](https://www.kaggle.com/datasets/omershafiq/bitcoin-network-transactional-metadata/data) published by _Omar Shafiq_. The dataset was created for research on blockchain anomaly and fraud detection and consists of a directed-acyclic graph (DAG) which was created from the bitcoin network transaction data from 2011-2013.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F750990%2Fcb02209b063d6521f834feb05fc852d0%2FScreenshot%202019-11-23%20at%208.51.27%20PM.png?generation=1574535111225673&alt=media)

From this, a number of features of each transaction were gathered:

* tx_hash: Hash or unique id of the bitcoin transaction.
* indegree: Number of transactions that are inputs of tx_hash
* outdegree: Number of transactions that are outputs of tx_hash.
* in_btc: Number of bitcoins on each incoming edge to tx_hash.
* out_btc: Number of bitcoins on each outgoing edge from tx_hash.
* total_btc: Net number of bitcoins flowing in and out from tx_hash.
* mean_in_btc: Average number of bitcoins flowing in for tx_hash.
* mean_out_btc: Average number of bitcoins flowing out for tx_hash. 
* in-malicious: Will be 1 if the tx_hash is an input of a malicious transaction.
* out-malicious: Will be 1 if the tx_hash is an output of a malicious transaction.
* is-malicious: Will be 1 if the tx_hash is a malicious transaction. (based on [https://www.kaggle.com/omershafiq/bitcoin-hacks-2010to2013](https://www.kaggle.com/omershafiq/bitcoin-hacks-2010to2013))
* out_and_tx_malicious: Will be 1 if the tx_hash is a malicious transaction or an output of a malicious transaction.
* all_malicious: Will be 1 if the tx_hash is a malicious transaction or an output of a malicious transaction or input of a malicious transaction.

### 3.1 Problems with this dataset

**3.1.1 Imbalanced**
The dataset is heavily skewed in favour of non-malicious transactions as is often the case with anomaly detection problems; from over 30 million transactions in the dataset, only 108 have been confirmed to be malicious. The effects not only the imbalance in the dataset, but also just the sheer lack of fraud samples will make traditional supervised neural network training very difficult.

**3.1.3 Potential for label noise**
Label noise refers to examples that belong to one class that are assigned to another class. For example, the dataset includes Bitcoin transactions. Given the properties of bitcoin transactions, it is not an unreasonable assumption that there are probbaly many transactions that have been used for fraudulent activities that have not been labelled as such. This label noise can cause problems when trying to classify between the two types of transactions because the model could think it's learning features for a legitimate transaction, but in fact due to a mislabelling (from the human researcher), it is actually learning features from a fraudulent transaction. 

For imbalnced datasets, this can have an even more pronounced effect. Given that examples in the positive class are so few, losing some to noise reduces the amount of information available about the minorty class even further.

Of couse, this is just speculation as we cannot be certain if the dataset includes already mislabled positive or negative samples, but it is something that should be taken into consideration when evaluating the performance of the models.

## 4. Feature Extraction & Engineering

In order to create an appropriate dataset to use in training, I chose to use a random sample of around 10000 non-malicious transaction data from the original dataset and combined them with all 108 malicious transactions to form the experimental dataset.

In addition to this, I used a publicly available APIs by [blockchain.com](https://www.blockchain.com/explorer/api/blockchain_api) to query the Bitcoin blockchain in order to gather further information about the transactions including:

* Size: The size of the transaction in bytes (used to calculate the fee proportion)
* Transaction fee: In satoshis
* Time: Timestamp of transaction in milliseconds since epoch

To further add meaningful data which the machine learning model will learn from, I decided to create a new feature from these existing ones, with the goal of capturing the patterns associated with the types of anomalies described earlier in this paper including:

* sat_per_byte (the transaction fee per byte of the transaction - indicates how relatively ‘expensive’ the transaction is)

## 5. Evaluating the Models

Accuracy is not an appropriate metric for evaluating fraud detection models because it does not fully represent the true performance of the model's predictive capabilities. Take for example a dataset consisting of 1000 transactions, 950 of which are legitimate, whilst 50 being fraudulent. A binary classification model that classifies all transactions as legitimate can achieve a accuracy score of 95% for the entire dataset inspite of having an accuracy of 0% for fraudulent transactions!

Two important metrics that provide a better measure of model performance are *precision* and *recall*. Precision describes how accurate the model was when it made a prediction that a data point was positive; measured by the true positives (TP), divided by the TP plus the false positives (FP). 
Recall is the measure of TP divided by the TP plus the false negatives (FN) - describing how much of the total positive classes the model was able to predict (regardless of how many times it got a prediction wrong).

Depending on the cost, or consequence of getting a prediction wrong (either way), the model should be optimised to produce a high recall or high precision. In cases where the cost of false negatives is very high, meaning classifying a positive class as negative, then recall is the more important metric because missing a potential positive case is worse than mis-identifying negative cases as positive. Optimising for precision on the other hand may be more appropriate in cases when he cost of such failure is low, or to reduce human workload. There are several methods to optimise for precision or recall, the manner in which a threshild is set cn be used to reflect the precision and recall preferences for each specific use case.

Another important metric for measuring the performance of fraud detection model is ROC curve (receiver operating characteristic curve). This measures the TP rate divided by the FP rate over many classifying thresholds. 

The ROC curve illustrates the trade-off between the recall (TP rate) or how many negative samples the model can correctly identify, and the false positive rate (FP rate) or the rate at which the model incorrectly idetifies a negative sample as being positive, at various thresholds.

The AUC metric, or Area Under the Curve. This provides an aggregate measure of performance across all possible thresholds. The AUC is a single scalar value that quantifies the overall performance of a binary classification model as represented by its ROC curve.

Finally, there is the F1 score which describes the tradeoff between precision and recall. Higher f1 scores *usually* indicate better performance, although this depends on the context. For our purposes, although recall is the most important metric, it should not be maximised at the expense of a very poor precision (otherwise our model will be mislabelling far too many legit transactions as fraudulent), and thus f1 seems to be an appropriate metric to measure for the experiments.

All of these will be utilised to measure the performance of all my models in this project.

* The AUC gives us an overall, easy to understand metric for evaluating the model's overall performance.
* The ROC curve gives us a good idea of how well the model can identify the fraudulent transactions. It can also be used to identify the best threshold for maximising labelling of fraudulent transactions, while mimimising mislabelling of legit transacions as fraudulent, which could be used as a threshold in future predictions.
* The Recall and precision give more detailed understanding of the behaviour of the model and how it is classifying samples. They can be analysed to identify the presence of majoriy bias for example.

For this project, we will say that the cost of missing a fraudulent transaction is moderate/high but not *extremely* high while the cost of mislabelling a legit transaction as fraudulent as relatively low. Why? Because once a Bitcoin transaction is confirmed by the network, it cannot be stopped regardless of whether it is used for fraudulent activity or not. Classification of fraudulent Bitcoin transactions is only useful *after the fact* and so such algorithms would most likely be used as an aid to help in human investigation into fraudulent activities involving Bitcoin, rather than being deployed as a preventative measure. Further, by labelling a legitimate transaction as fraudulent, the sender of the Bitcoin is not affected in any way since the flagging og their transaction is made outside of the Bitcoin network and so as far as theu're concerned, it little difference (unless however as a result they are then put on under further investigation, which is outside of the scope of this project). Therefore the machine learning model should be optimised to favour producing a higher recall score as mislabelling a legit transaction doesn't have as much consequence.

## 6. Project Methodology
**MLP/shallow neural network:**
If approaching this problem from a binary classification angle, we can train a simple neural network such as a MLP (multi-layer perceptron) to categorise our data into two classes, legit or fraud. Since the availability of fraud data is limited, we can try and balance out the dataset by using a few techniques: 1) undersampling the legit transaction data 2) adding class weights to make the model pay more attention to fraud classes during training 3) synthetically oversample the fraud transaction data to make the number of fraud and legit the same. Experiments will be ran for each of these. Additionally, experiments will be run with different MLP hyperparamerers to try and get the best training and test result.

**Autoencoder:**
If approaching this from a anomaly detection standpoint the autoencoders will be trained on legitimate transactions only. The idea is that the autoencoder will learn from these legitimate transactions and get good at reconstructing legitimate transactions from the latent code to produce a low reconstruction error. When the autoencoder encounters a fraudulent transcation, the reconstruction error should be significantly larger - indicating an anomalous or fraudulent transaction.

**Unsupervised learning methods**
If time permits, I will perform some unsupervised learning techniques such as isolation forest and DBSCAN in order to compare the results with supervised methods explained above.


## 7. Experimental Results
For this project, 2 supervised (MLP and Autoencoder) and 2 unsupervised methods (Isolation forest and DBSCAN) were used to detect fraudulent transactions.

MLP:
The model with the best f1 score was trained on sythetically oversampled minority data, resulting in an f1 of 30%. However, there was arguably a better model in which although the f1 score was 17%, the recall was much higher (82%) compared with only 45%.

Autoencoder:
The experimental results did not produce any applicable results, with the highest recall being only 23% for the best experiment.

Isolation Forest:
Isolation forest produced teh highest f1 score of all experiments (56%), indicating a balance between precision and recall. Despite this, the MLP experiment producing f1 of 17% can be seen as better for fraud detetcion as it produced 82% recall compared with only 57% of this isolation forest model.

DBSCAN:
DBSCAN model was only able to successfully identify 1/108 fraud transactions and did not produce any significant results

## 8. Further Work
Due to the time and spec limitations, there are many things that I would like to explore in the future. Firstly, work can be done in the feature engineering section. As seen in other work on this topic, features such as time intervals between receiving and spending transactions could provide the model with more accurate features to learn from. Also, the dataset contained fraudulent transactions from between 2011-2013. If more data is available, there will be a much larger pool of fraudulent transactions and since these fraud transactions do not always have a common structure, it may help get better results.

Further, I would introduce a standardised research experiment metholdology. i.e. make sure all experiments are carried out in a set method by keeping all variables the same, just changing the model used.

## 9. Conclusions
In this study, I focused on the detection of Bitcoin fraud transactions using supervised machine learning methods. I collected historical Bitcoin transaction data and extracted 11 features based on the characteristics of fraud transactions. I used two supervised methods—the MLP and Autoencoder methods to classify the features. The experimental results showed that the MLP have better classification abilities than the other approaches. Further, I performed oversampling to equalize the unbalanced training set. The experiments showed that the recall was further improved by equalization.

My next research directions include two aspects: (1) extracting more targeted features for theft transactions and combining multiple machine learning algorithms to improve the detection results and (2) exploring graph based neural network architectures.


## 10. References: 
 - Pang, G., Shen, C., Jin, H., & Hengel, A. van den. (2023, June 5). Deep Weakly-supervised Anomaly Detection. ArXiv.org. https://doi.org/10.48550/arXiv.1910.13601
 - Deep Learning for Anomaly Detection. (n.d.). Ff12.Fastforwardlabs.com. https://ff12.fastforwardlabs.com/
 - Chen, B., Wei, F., & Gu, C. (2021). Bitcoin Theft Detection Based on Supervised Machine Learning Algorithms. Security and Communication Networks, 2021, 1–10. https://doi.org/10.1155/2021/6643763
 - Shafiq, O. (2019). ANOMALY DETECTION IN BLOCKCHAIN.
 - Omer Shafiq. (2019). Bitcoin Hacked Transactions 2010-2013. IEEE Dataport. https://dx.doi.org/10.21227/7f0c-df28
 - Bitcoin Network Transactional Metadata. (n.d.). Www.kaggle.com. Retrieved December 13, 2023, from https://www.kaggle.com/datasets/omershafiq/bitcoin-network-transactional-metadata
