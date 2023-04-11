plt.figure(figsize=(10,6))
plt.hist(df['strike'], edgecolor = 'green')
plt.xlabel('strike price')
plt.title("strike prices")
plt.show()


plt.figure(figsize = (12,10))
sns.scatterplot(df['openInterest'], df['volume'], hue=df['optionType'])
plt.figure(figsize=(15,8))
sns.boxplot(x=df['impliedVolatility'])



plt.figure(figsize=(50,35))
sns.pairplot(df[['strike', 'ask', 'impliedVolatility', 'optionType', 'volume','percentChange','TimeToExpiration']], diag_kind='kde')

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [strike]')
plt.ylabel('Predictions [strike]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)

