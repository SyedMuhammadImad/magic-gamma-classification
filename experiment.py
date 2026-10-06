"""Reproducible academic experiment; all transforms fit training data only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
def classification(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(y,p).tolist(),'report':classification_report(y,p,output_dict=True,zero_division=0)}
def neural(x,y,vx,vy,tx,regression=False):
    import torch,copy
    from torch import nn
    torch.set_num_threads(2);torch.manual_seed(42)
    x,vx,tx=[torch.tensor(z,dtype=torch.float32) for z in (x,vx,tx)]
    scale=max(float(np.std(y)),1.) if regression else 1.
    mean=float(np.mean(y)) if regression else 0.
    y,vy=[torch.tensor((z-mean)/scale,dtype=torch.float32).reshape(-1,1) if regression else torch.tensor(z,dtype=torch.long) for z in (y,vy)]
    model=nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1 if regression else 2))
    criterion=nn.MSELoss() if regression else nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    best=float('inf');weights=None;wait=0;history=[]
    for epoch in range(100):
        model.train()
        for batch in torch.randperm(len(x)).split(128):
            optimizer.zero_grad();loss=criterion(model(x[batch]),y[batch]);loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode(): score=float(criterion(model(vx),vy))
        history.append(score)
        if score<best-1e-5:best=score;weights=copy.deepcopy(model.state_dict());wait=0
        else:wait+=1
        if wait>=12:break
    model.load_state_dict(weights)
    with torch.inference_mode():out=model(tx)
    pred=out.numpy().ravel()*scale+mean if regression else out.argmax(1).numpy()
    return pred,{'epochs':len(history),'best_validation_loss':best,'target_scale':scale,'target_mean':mean}
def experiment(data):
    columns=['flength','fwidth','fsize','fconc','fconc1','fasym','fm3long','fm3trans','falpha','fdist','class']
    frame=pd.read_csv(data,names=columns).drop_duplicates()
    if not set(frame['class'])<={'g','h'}:raise ValueError('Labels must be g/h')
    x=frame.iloc[:,:10].to_numpy(float);y=(frame['class']=='g').to_numpy(int)
    if not np.isfinite(x).all():raise ValueError('Nonfinite features')
    a,b,ya,yb=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    a,v,ya,yv=train_test_split(a,ya,test_size=.25,random_state=42,stratify=ya)
    scaler=StandardScaler().fit(a);a,v,b=[scaler.transform(z) for z in (a,v,b)]
    rng=np.random.default_rng(42);counts=np.bincount(ya);indices=np.concatenate([rng.choice(np.flatnonzero(ya==label),size=max(counts),replace=True) for label in [0,1]])
    ax,ay=a[indices],ya[indices]
    models={'KNN':KNeighborsClassifier(n_neighbors=3),'NaiveBayes':GaussianNB(),'LogisticRegression':LogisticRegression(max_iter=1000),'SVM':SVC()}
    results={}
    for name,model in models.items():model.fit(ax,ay);results[name]=classification(yb,model.predict(b))
    prediction,training=neural(ax,ay,v,yv,b);results['NeuralNetwork']=classification(yb,prediction);results['NeuralNetwork']['training']=training
    return {'dataset':'UCI MAGIC Gamma Telescope','seed':42,'train':len(a),'validation':len(v),'test':len(b),'balanced_training_rows':len(ax),'models':results}
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('metrics.json'))
    args=parser.parse_args()
    results=experiment(args.data)
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results))
