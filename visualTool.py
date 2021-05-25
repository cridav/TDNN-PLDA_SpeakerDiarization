# Quick class to visualize some results
# by: @cridav

class visualTool():
  def logHTML(self, Id, value):
    """
    Transforms a plotly image to HTML format and then uploads
    it to WANDB under the given Id"""
    wandb.log({str(Id): wandb.Html(plotly.io.to_html(value))})
  
  def log(self, Id, value):
    """
    Uploads a plotly Image into WANDB under the given Id"""
    wandb.log({str(Id): value})
    

  def binaryLabelCreator(self, labels, prediction):
    """
    One vs. All label-prediction evaluation
    returns two tuples of three elements each
    first tuple: flatten label (#classes * elements) of label, prediction and score (two classes 1 = correct, 0 = incorrect)
    second tuple: array [#classes, elements] for label, prediction and score (One vs. All)
    eg:
    binaryLabelCreator(np.array([2,2,3]), np.array([2,3,3]))
    (
      (array([1., 1., 0., 0., 0., 1.]),
      array([1., 0., 0., 0., 1., 1.]),
      array([1., 0., 1., 1., 0., 1.])),
      (array([[1., 1., 0.],
            [0., 0., 1.]]),
      array([[1., 0., 0.],
            [0., 1., 1.]]),
      array([[1., 0., 1.],
            [1., 0., 1.]]))
    )
    """
    dictPredicts = dict()
    uniqueLabels = np.unique(labels)
    binaryLabel = np.zeros(shape = (len(uniqueLabels), len(labels)))
    binaryPreds = np.zeros(shape = (len(uniqueLabels), len(labels)))
    predsScore = np.zeros(shape = (len(uniqueLabels), len(labels)))
    for indx, clss in enumerate(uniqueLabels):
      binaryLabel[indx, :] = np.where(labels == clss,1,0)
      binaryPreds[indx, :] = np.where(prediction == clss,1,0)
      predsScore [indx, :] = ( binaryLabel[indx, :] == binaryPreds[indx, :]).astype(int) 

    binaryLabel_ = np.concatenate(binaryLabel, axis = 0)
    binaryPreds_ = np.concatenate(binaryPreds, axis = 0)
    predsScore_ = np.concatenate(predsScore, axis = 0)

    return (binaryLabel_, binaryPreds_, predsScore_), (binaryLabel, binaryPreds, predsScore)

  def plotROC(self, fpr, tpr):
    """
    fpr = false positive rate
    tpr = true positive rate
    returns fig
    """
    fig = px.area(x=fpr, y=tpr)
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, ticks = 'inside', showline=True, linewidth=2, linecolor='black',
                  showgrid = True, gridwidth = 0.5, gridcolor = 'black', dtick = 0.2)
    fig.update_xaxes(constrain='domain',
                  ticks = 'inside',showline=True, linewidth=2, linecolor='black',
                  showgrid = True, gridwidth = 0.5, gridcolor = 'black', dtick = 0.2)
    fig.update_layout(
        title={
            'text': f'ROC Curve (AUC={metrics.auc(fpr, tpr):.4f})',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        font_color = 'black',
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        width=500, height=500
    )
    return fig

  def plotConfMatrix(self, cm, matrixTitle = 'Confusion Matrix'):
    """
    Plots the confusion matrix of an array of arrays:
    cm:
  [[31  2  0  0]
  [ 0 16  0  0]
  [ 0  0 22  0]
  [ 1  1  1 22]]

  cm can be obtained using sklearn
  cm = metrics.confusion_matrix(labels,prediction)
    """
    fig = ff.create_annotated_heatmap(cm[::-1,:])
    # add title
    fig.update_layout(title={
                      'text': matrixTitle,
                      'y':0.85,
                      'x':0.5,
                      'xanchor': 'center',
                      'yanchor': 'top'
                      },
                      font_color = 'black',
                      plot_bgcolor = 'rgba(0, 0, 0, 0)',
                      paper_bgcolor = 'rgba(0, 0, 0, 0)',
                      width=500, height=500
                    )

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            # x=0.5,
                            y=-0.1,
                            showarrow=False,
                            text="Predicted label",
                            xref="paper",
                            yref="paper",
    ))
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.1,
                            # y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            # align = 'left'
                            xref="paper",
                            # yref="paper"
                            ))

    # add colorbar
    fig['data'][0]['showscale'] = True
    return fig

  def plotPCA2(self, embd_pca2, labels, pcaTitle = "Cluster visualization PCA-2"):
    prediction = edict({'component1':[],'component2':[],'component3':[],'speaker':[]})
    prediction.component1 = embd_pca2[:,0]
    prediction.component2 = embd_pca2[:,1]
    prediction.speaker = labels
    pdPrediction = pd.DataFrame(prediction, columns = ['component1','component2','speaker'])
    pdPrediction['markerSize'] = 1
    # print(pdPrediction)

    fig1 = px.scatter(pdPrediction,
                    x = 'component1',
                    y = 'component2',
                    color = 'speaker',
                    size = 'markerSize',
                    symbol = 'speaker',)

    fig1.update_layout(
          font_color = 'black',
          paper_bgcolor = 'rgba(0, 0, 0, 0)',
          height=600, width=600,
          title={
                      'text': str(pcaTitle),
                      'y':1,
                      'x':0.5,
                      'xanchor': 'center',
                      'yanchor': 'top'
                      },
      )
    # hide colorscale
    fig1.update_coloraxes(showscale=False)
    fig1.update_yaxes(
                      showline=True,
                      linewidth=2,
                      mirror=True,
                      )
    fig1.update_xaxes(
                      showline=True,
                      linewidth=2,
                      mirror=True,
                      )
    return fig1

  def plotPCA3(self, embd_pca3, labels, pcaTitle = "Cluster visualization PCA-3"):
    """
    Plots a 3D scatter for the PCA obtained from the xvectors.
    PCA can be obtained using:
      from sklearn.preprocessing import Normalizer
      from sklearn.decomposition import PCA

      # Normalization
      transformer = Normalizer()
      embeddings = transformer.fit_transform(embeddings)
      # PCA
      pca = PCA(n_components=3)
      embd_pca3 = pca.fit_transform(embeddings)
    """
    prediction = edict({'component1':[],'component2':[],'component3':[],'speaker':[]})
    prediction.component1 = embd_pca3[:,0]
    prediction.component2 = embd_pca3[:,1]
    prediction.component3 = embd_pca3[:,2]
    prediction.speaker = labels
    pdPrediction = pd.DataFrame(prediction, columns = ['component1','component2','component3','speaker'])
    pdPrediction['markerSize'] = 1
    fig2 = px.scatter_3d(pdPrediction,
                    x = 'component1',
                    y = 'component2',
                    z = 'component3',
                    color = 'speaker',
                    size = 'markerSize',
                    symbol = 'speaker',
                    )

    fig2.update_layout(
          font_color = 'black',
          plot_bgcolor = 'rgba(0, 0, 0, 0)',
          paper_bgcolor = 'rgba(0, 0, 0, 0)',
          height=600, width=600,
          title={
                      'text': str(pcaTitle),
                      'y':0.9,
                      'x':0.5,
                      },
          legend=dict(x=1,y=0.8),
      )
    fig2.update_coloraxes(showscale=False)
    return fig2


vt = visualTool()