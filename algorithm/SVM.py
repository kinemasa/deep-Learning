##iris(アヤメの情報のデータセット)
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data  
y = iris.target

##訓練用とテストデータの分割
##random ステートは乱数生成のシード
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

##正規化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


##次元圧縮目的の主成分分析
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

##SVMの実装
from sklearn import svm
model = svm.SVC()
model.fit(X_train_pca, y_train)

from sklearn.metrics import accuracy_score
# テストデータに対してもPCA変換を実施
X_test_pca = pca.transform(X_test)
# 予測
y_pred = model.predict(X_test_pca)
# 評価
accuracy_score(y_test, y_pred)


from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
 
# 決定境界の描画
DecisionBoundaryDisplay.from_estimator(model,
                                       X_train_pca,
                                       plot_method='contour',
                                       cmap=plt.cm.Paired,
                                       levels=[-1, 0, 1],
                                       alpha=0.5,
#                                        linestyles=['--', '-', '--'],
                                       xlabel='first principal component',
                                       ylabel='second principal component',
                                       )
 
# 学習データの描画
for i, color in zip(model.classes_, 'bry'):
    idx = np.where(y_train == i)
    plt.scatter(
            X_train_pca[idx, 0],
            X_train_pca[idx, 1],
            c=color,
            label=iris.target_names[i],
            edgecolor='black',
            s=20,
    )
 
#
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors='none',
    edgecolors='k')

plt.show()

