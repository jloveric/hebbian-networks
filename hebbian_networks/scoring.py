"""Fixed individual-layer and cumulative class scores from the paper."""
import numpy as np

def score_breakdown(energy,normalized,labels):
    """Only the score changes; all activities come from one ordinary forward."""
    assert energy.shape==normalized.shape and energy.shape[:2]==(len(labels),10)
    def metric(scores,layers):
        prediction=scores.argmax(1)
        errors=int(np.count_nonzero(prediction!=labels))
        return dict(layers=layers,images=len(labels),errors=errors,accuracy=1-errors/len(labels)),prediction
    individual=[];prefix=[];individual_predictions=[];prefix_predictions=[]
    for layer in range(energy.shape[2]):
        row,prediction=metric(energy[:,:,layer],[layer+1])
        individual.append(row);individual_predictions.append(prediction)
    # Positive scalar normalization cannot change a single layer's ranking.
    prefix.append(dict(individual[0]));prefix_predictions.append(individual_predictions[0])
    for end in range(2,energy.shape[2]+1):
        row,prediction=metric(np.sum(normalized[:,:,:end],axis=-1,dtype=np.float32),list(range(1,end+1)))
        prefix.append(row);prefix_predictions.append(prediction)
    return dict(individual=individual,prefix=prefix),np.stack(individual_predictions,-1),np.stack(prefix_predictions,-1)
