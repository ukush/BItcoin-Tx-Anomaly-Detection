# Binary classification of anomalous Bitcoin transactions using supervised machine learning


### The aim of this project is to develop a machine learning model which can classify anomalous transactions on the Bitcoin Network using a combination of various machine learning techniques and neural network architectures. 

**Introduction**

Since the inception of Bitcoin in 2009, there have been concerns with regards to its security and the potential for it to be used in illegal activity, partly due to its pseudo-anonymous properties. Transactions over the Bitcoin network are not directly linked to personal identities, although this can be established through deeper investigation as more often than not, the Bitcoins are purchased using personal bank accounts on centralised exchanges, which often require proof of identity. 

Nevertheless, Bitcoin fraud is still an inevitability. For those who know how to hide their identity, Bitcoin provides a better alternative to traditional currencies for malicious purposes. Dark web marketplaces such as the infamous Silk Road heavily relied on the use of cryptocurrencies such as Bitcoin in order to facilitate the transfer of illegal weapons and drugs amongst users.

Over the last decade, there have been multiple high profile cases of centralised exchanges being hacked, resulting in the leaking of thousands of Bitcoin private keys - which give a user access to their Bitcoin funds. Once leaked, hackers can use these private keys to transfer funds from these addresses to another address, one which only the hackers have the private key for. Since Bitcoin transactions are non-reversible, there is no way for customers to retrieve their funds.

The field of anomaly detection is a topic which has been dealt with extensively, especially in the financial sector. Now with the development of machine learning programming frameworks such as tensorflow, keras and opencv, the barrier to entry is much lower and therefore far more work in this field can be achieved.

This project aims to explore the potential uses of machine learning to detect anomalous transactions made over the Bitcoin network.


### Background

**Bitcoin**

Bitcoin is a decentralised, peer-to-peer money ecosystem which allows participants to send and receive value using nothing more than a computer and an internet connection. In the Bitcoin network, transactions are made directly between participants, without the need for a middle-man. Bitcoin transactions are final and cannot be reversed. It is therefore extremely important that users are always in control of the bitcoins they own. This ownership is proven through special ‘keys’ which allow the owners to ‘unlock’ bitcoin funds which have been sent to them. Therefore, whoever controls the keys, controls the bitcoins. 

**Transaction Inputs and outputs**


![Overview of Bitcoin Transaction Inputs and Outputs](https://en.bitcoin.it/w/images/en/f/f1/Bitcointransactions.JPG)



In a bitcoin transaction, there is nothing being ‘sent’ per say. Rather, the owner of a certain number of bitcoins specifies who the new owner will be. They first ‘lock’ these funds using a special locking script and a unique digital signature. These funds can now only be unlocked, or spent, by the user that owns a particular key, a key that the old owner specifies. This forms the _output _of a transaction. The outputs of a previous transaction become part of the input to the next transaction when a new owner wants to spend the funds that were sent to them by unlocking the unspent transaction outputs (UTXO) using their special key.

**Anomalous Transactions**

**What is an anomalous transaction?**

An anomalous transaction is one that contains data points which are significantly different from those of a normal transaction. This can come in many forms.** **This paper describes some of the typical types or categories of anomalous transactions:

**Bitcoin theft**

This usually occurs when a malicious actor gains access to a victim’s private key, giving them complete control over the Bitcoin funds in the victim’s wallet. This can be a result of a breach or hack of a centralised cryptocurrency exchange in which the user's information is leaked; or it can be a result of a scam where victims are tricked into giving scammers access to their wallets voluntarily. Regardless of the method of acquisition, these types of thefts have some common patterns.



* Victim’s wallet is quickly emptied, with all funds being transferred to the thief’s wallet in a **single transaction output**.
* Fraudulent transactions may involve **abnormal fees in relation to the transaction amount. **Miners will prioritise transactions with large fees and so those trying to transfer stolen funds as quickly as possible may introduce large fees for these transactions.

**Laundering or other malicious activities**

As mentioned, Bitcoin is a better digital currency alternative than traditional currencies to use for illegal activities because of the pseudo-anonymity it provides, as well as its peer-to-peer nature. Thus, it is used by criminal, drug cartels and crime syndicates for cross-border transactions for purposes of laundering illegal money. It is worth noting that there are far better cryptocurrencies for this purpose such as privacy coins like Monero, where transactions are completely untraceable and anonymous; however Bitcoin is still used for illegal activities due to its popularity in relation to other cryptocurrencies. 

Pattern of money laundering may include:



* Tainting of bitcoins whereby some possibly **stolen funds are mixed with other, legal funds** and used together as inputs to a new transaction, thus convoluting the transaction making it more difficult to detect.
* Thief's may try to mask their actions by making **numerous, smaller transaction outputs** **in a short space of time** in an attempt to confuse the victim and authorities
* Making transactions at **unusual times of the day** to avoid detection.

**Crypto Ransom & Extortion**

**Bitcoin is also a popular currency for **perpetrators of ransoms and exploitation. A characteristic of these types of crimes are that criminals will often force victims to make a** single or a few large transactions to a single address** or potentially make smaller, **more numerous transactions to different addresses**, again in order to confuse investigators after the fact.

**Network Attacks**

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

**Problems with this description**

An issue with categorising malicious activity using transaction markers is that there will undoubtedly be scenarios where the characteristics described may be associated with normal transaction behaviour:

**Numerous Small Transactions:**



* Microtransactions: Some businesses or applications rely on microtransactions, where small amounts of cryptocurrency are transferred frequently. This is common in gaming or content platforms.
* Dollar-Cost Averaging: Investors might engage in dollar-cost averaging, making frequent small purchases of cryptocurrency over time.

**Higher Fees:**



* Urgency or Priority Transactions: Users might intentionally choose higher fees to ensure quicker confirmation times for their transactions, especially during periods of network congestion.
* Complex Smart Contract Execution: Certain transactions involving complex smart contracts or advanced features may require higher fees.

**Large Outputs to Single Address:**



* Exchange Withdrawals: When users withdraw funds from a cryptocurrency exchange, a large output to a single address is common.
* Business Transactions: Large payments between businesses or entities can result in a single, large output.

**Numerous Small Outputs to Multiple Addresses:**



* Payment Splitting: In a business context, payments might be split into numerous smaller outputs to distribute funds among multiple recipients.
* Wallet Shuffling: Some users intentionally split their funds into smaller amounts and distribute them across multiple addresses for privacy reasons.

**Late Hours of the Day:**



* Global Nature of Cryptocurrency Markets: Participants in cryptocurrency markets are distributed globally, leading to transactions occurring at various times, including late hours.

Regardless, the transaction markers described can still provide a useful indication of potential suspicious or malicious activity.


### Dataset Description

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

The dataset is heavily skewed in favour of non-malicious transactions as is often the case with anomaly detection problems; from over 30 million transactions in the dataset, only 108 have been confirmed to be malicious. Thus, common anomaly detection methods will need to be used in order to sufficiently extract meaningful features from the dataset to be used in the training of the machine learning model.

**Assumptions and goals**

For the purposes of this project, the classification of anomalous transactions will be considered from a_ transaction structure_ perspective. This means features relating to the structure of the transaction itself will be used to train the machine learning model, not features from a network or user level. This is mainly due to the dataset that is available and due to the time constraints of the project. If I had access to the entire Bitcoin blockchain (perhaps by running my own node), a deeper level of investigation into methods of tracking fraudulent transactions could have been explored, such as the ones in [this paper](https://pure.hw.ac.uk/ws/portalfiles/portal/24387537/Conf_paper.pdf). It is also important to define a single goal of this project, as opposed to convoluting it with multiple outcome goals. 

The model will attempt to learn from the data and detect the transactions that it thinks are anomalous. Anomalous transactions are defined as those that exhibit characteristics similar to those discussed earlier in this paper. Therefore the goal of this model is not to detect particular types of transactions e.g. only theft transactions or only network attacks, but all of these.

**Feature Extraction & Engineering**

In order to create an appropriate dataset to use in training, I chose to use a random sample of around 10000 non-malicious transaction data from the original dataset and combined them with all 108 malicious transactions to form the experimental dataset.

In addition to this, I used a publicly available APIs by [blockchain.com](https://www.blockchain.com/explorer/api/blockchain_api) to query the Bitcoin blockchain in order to gather further information about the transactions including:



* Size: The size of the transaction in bytes (used to calculate the fee proportion)
* Transaction fee: In satoshis
* Time: Timestamp of transaction in milliseconds since epoch

To further add meaningful data which the machine learning model will learn from, I decided to create a new feature from these existing ones, with the goal of capturing the patterns associated with the types of anomalies described earlier in this paper including:



* sat_per_byte (the transaction fee per byte of the transaction - indicates how relatively ‘expensive’ the transaction is)

**Machine learning methods for anomaly detection**

**Proposed neural network types**

MLP

DNN

CNN

Autoencoders

Bayesian Networks
