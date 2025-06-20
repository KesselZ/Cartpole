COMMENT

STDP + RL weight adjuster mechanism

Original STDP code adapted from:
http://senselab.med.yale.edu/modeldb/showmodel.asp?model=64261&file=\bfstdp\stdwa_songabbott.mod

Adapted to implement a "nearest-neighbor spike-interaction" model (see
Scholarpedia article on STDP) that just looks at the last-seen pre- and
post-synaptic spikes, and implementing a reinforcement learning algorithm based
on (Chadderdon et al., 2012):
http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0047251

Example Python usage:

from neuron import h

## Create cells
dummy = h.Section() # Create a dummy section to put the point processes in
ncells = 2
cells = []
for c in range(ncells): cells.append(h.IntFire4(0,sec=dummy)) # Create the cells

## Create synapses
threshold = 10 # Set voltage threshold
delay = 1 # Set connection delay
singlesyn = h.NetCon(cells[0],cells[1], threshold, delay, 0.5) # Create a connection between the cells
stdpmech = h.STDP(0,sec=dummy) # Create the STDP mechanism
presyn = h.NetCon(cells[0],stdpmech, threshold, delay, 1) # Feed presynaptic spikes to the STDP mechanism -- must have weight >0
pstsyn = h.NetCon(cells[1],stdpmech, threshold, delay, -1) # Feed postsynaptic spikes to the STDP mechanism -- must have weight <0
h.setpointer(singlesyn._ref_weight[0],'synweight',stdpmech) # Point the STDP mechanism to the connection weight

Version: 2013oct24 by cliffk

ENDCOMMENT

NEURON {
  POINT_PROCESS STDP : Definition of mechanism
  POINTER synweight : Pointer to the weight (in a NetCon object) to be adjusted.
  : LTP/LTD decay time constants (ms) for Hebbian (pre-before-post-synaptic spikes), and anti-Hebbian (post-before-pre-synaptic) cases.
  RANGE tauhebb, tauanti
  : Max adjustment (positive or negative) for Hebbian and anti-Hebbian cases (i.e., as inter-spike interval approaches zero)
  : Set positive for LTP and negative for LTD.
  RANGE hebbwt, antiwt
  : Maximum interval between pre- and post-synaptic events for an starting an eligibility trace.
  : There are separate ones for the Hebbian and anti-Hebbian events.
  RANGE RLwindhebb, RLwindanti
  : Use exponentially decaying eligibility traces?  If 0, eligibility traces are binary, turning on at beginning and off after time has
  : passed corresponding to RLlen.
  RANGE useRLexp
  : Length of eligibility Hebbian and anti-Hebbian eligibility traces, or decay time constants if traces are decaying exponentials
  RANGE RLlenhebb, RLlenanti
  : Max synaptic weight adjustments based on reward or punishing signal by Hebbian and anti-Hebbian eligibility traces.
  RANGE RLhebbwt, RLantiwt
  RANGE wbase, wmax : The maximum,minimum weight for the synapse
  RANGE softthresh : Flag turning on "soft thresholding" for the maximal adjustment parameters.
  RANGE STDPon : Flag for turning STDP adjustment on / off.
  RANGE RLon : Flag for turning RL adjustment on / off.
  RANGE verbose : Flag for turning off prints of weight update events for debugging.
  RANGE tlastpre, tlastpost : Remembered times for last pre- and post-synaptic spikes.
  RANGE tlasthebbelig, tlastantielig : Remembered times for Hebbian anti-Hebbian eligibility traces.
  RANGE interval : Interval between current time t and previous spike.
  RANGE deltaw
  RANGE newweight
  RANGE skip : Flag to skip 2nd set of conditions
  RANGE cumreward : cumulative reward magnitude so far
  RANGE maxreward : max reward for scaling
  GLOBAL initialtime
  :RANGE NOSTDPTAG
}

ASSIGNED {
  synweight
  tlastpre   (ms)
  tlastpost  (ms)
  tlasthebbelig   (ms)
  tlastantielig  (ms)
  interval    (ms)
  deltaw
  newweight
  cumreward
}

INITIAL {
  reset_eligibility()
}

PARAMETER {
  tauhebb  = 10  (ms)
  tauanti  = 10  (ms)
  hebbwt =  1.0
  antiwt = -1.0
  RLwindhebb = 10 (ms)
  RLwindanti = 10 (ms)
  useRLexp = 0 : default to using binary eligibility traces
  RLlenhebb = 100 (ms)
  RLlenanti = 100 (ms)
  RLhebbwt =  1.0
  RLantiwt = -1.0
  wbase = 0
  wmax  = 15.0
  softthresh = 0
  STDPon = 1
  RLon = 1
  verbose = 0
  skip = 0
  maxreward = 0
  : cumreward = 0
  :NOSTDPTAG = 0
  initialtime = 1000 (ms) : initialization time before any weight changes possible
}

PROCEDURE reset_eligibility () {
  tlastpre = -1            : no spike yet
  tlastpost = -1           : no spike yet
  tlasthebbelig = -1      : no eligibility yet
  tlastantielig = -1  : no eligibility yet
  interval = 0
  cumreward = 0
}

NET_RECEIVE (w) {
     :LOCAL deltaw
  deltaw = 0.0 : Default the weight change to 0.
  skip = 0
  if (t < initialtime) {
    VERBATIM
    return;
    ENDVERBATIM
  }

  : if (verbose > 0)  { printf("t=%f (BEFORE) tlaspre=%f, tlastpost=%f, flag=%f, w=%f, deltaw=%f \n",t,tlastpre, tlastpost,flag,w,deltaw) }

  : Hebbian weight update happens 1ms later to check for simultaneous spikes (otherwise bug when using mpi)
  if ((flag == -1) && (tlastpre != t-1)) {
        skip = 1 : skip the 2nd set of conditions since this was artificial net event to update weights
        if (hebbwt != 0.0) {
          deltaw = hebbwt * exp(-interval / tauhebb) : Use the Hebbian decay to set the Hebbian weight adjustment.
          if (softthresh == 1) { deltaw = softthreshold(deltaw) } : If we have soft-thresholding on, apply it.
          adjustweight(deltaw) : Adjust the weight.
          :if (verbose > 0) { printf("Hebbian STDP event: t = %f ms; tlastpre = %f; w = %f; deltaw = %f\n",t,tlastpre,w,deltaw) } : Show weight update information if debugging on.
  }
  }
  : Ant-hebbian weight update happens 1ms later to check for simultaneous spikes (otherwise bug when using mpi)
  else if ((flag == 1) && (tlastpost != t-1)) { :update weight 1ms later to check for simultaneous spikes (otherwise bug when using mpi)
  skip = 1 : skip the 2nd set of conditions since this was artificial net event to update weights
  if (antiwt != 0.0) {
          deltaw = antiwt * exp(interval / tauanti) : Use the anti-Hebbian decay to set the anti-Hebbian weight adjustment.
          if (softthresh == 1) { deltaw = softthreshold(deltaw) } : If we have soft-thresholding on, apply it.
          adjustweight(deltaw) : Adjust the weight.
    :if (verbose > 0) { printf("anti-Hebbian STDP event: t = %f ms; deltaw = %f\n",t,deltaw) } : Show weight update information if debugging on.
  }
  }

  : If we receive a non-negative weight value, we are receiving a pre-synaptic spike (and thus need to check for an anti-Hebbian event,
  : since the post-synaptic weight must be earlier).
  if (skip == 0) {
    if (w >= 0) {
      interval = tlastpost - t  : Get the interval; interval is negative
      if  ((tlastpost > -1) && (-interval > 1.0)) { : If we had a post-synaptic spike and a non-zero interval...
        if (STDPon == 1) { : If STDP learning is turned on...
          :if (verbose > 0) {printf("net_send(1,1)\n")}
          net_send(1,1) : instead of updating weight directly, use net_send to check if simultaneous spike occurred (otherwise bug when using mpi)
  }
  : If RL and anti-Hebbian eligibility traces are turned on, and the interval falls within the maximum window for eligibility,
  : remember the eligibilty trace start at the current time.
        if ((RLon == 1) && (-interval <= RLwindanti)) { tlastantielig = t }
      }
      tlastpre = t : Remember the current spike time for next NET_RECEIVE.
    : Else, if we receive a negative weight value, we are receiving a post-synaptic spike
    : (and thus need to check for a Hebbian event, since the pre-synaptic weight must be earlier).
    }
    else {
      interval = t - tlastpre : Get the interval; interval is positive
      if  ((tlastpre > -1) && (interval > 1.0)) { : If we had a pre-synaptic spike and a non-zero interval...
        if (STDPon == 1) { : If STDP learning is turned on...
          :if (verbose > 0) {printf("net_send(1,-1)\n")}
          net_send(1,-1) : instead of updating weight directly, use net_send to check if simultaneous spike occurred (otherwise bug when using mpi)
        }
        if ((RLon == 1) && (interval <= RLwindhebb)) {
      : If RL and Hebbian eligibility traces are turned on, and the interval falls within the maximum window for eligibility,
      : remember the eligibilty trace start at the current time.
          tlasthebbelig = t
        }
      }
      tlastpost = t : Remember the current spike time for next NET_RECEIVE.
    }
  }
  : if (verbose > 0)  { printf("t=%f (AFTER) tlaspre=%f, tlastpost=%f, flag=%f, w=%f, deltaw=%f \n",t,tlastpre, tlastpost,flag,w,deltaw) }
}

PROCEDURE reward_punish (reinf) {
  :LOCAL deltaw
  :printf("RLon=%g,reinf=%g,fabs(reinf)=%g,fabs(reinf)>0.0=%g\n",RLon,reinf,fabs(reinf),fabs(reinf)>0.0)
  if (RLon == 1 && fabs(reinf)>0.0) { : If RL is turned on...
    :if(fabs(reinf) > 0.0){printf("reinf=%g\n",reinf)}
    deltaw = 0.0 : Start the weight change as being 0.
    if (RLhebbwt>0.0) {
      deltaw = deltaw + reinf * hebbRL() : If we have the Hebbian eligibility traces on, add their effect in.
    }
    if (RLantiwt<0.0) {
      deltaw = deltaw + reinf * antiRL() : If we have the anti-Hebbian eligibility traces on, add their effect in.
    }
    if (softthresh == 1) { : If we have soft-thresholding on, apply it.
      deltaw = softthreshold(deltaw)
    }
    :if (maxreward > 0.0) {
    :  if (cumreward > maxreward) {
    :    deltaw = 0.0
    :  } else {
    :    deltaw = deltaw * (1.0 - cumreward / maxreward)
    :  }
    :}
    :if(fabs(deltaw)>0.0 && tlasthebbelig>20 && t>2400 && cumreward>0){
    : printf("RL event: t = %g ms; reinf = %g; RLhebbwt = %g; RLlenhebb = %g; tlasthebbelig = %g; deltaw = %g, cumreward = %g\n",t,reinf,RLhebbwt,RLlenhebb,tlasthebbelig, deltaw, cumreward)
    :}
    :if (fabs(deltaw)>0.0){
      adjustweight(deltaw) : Adjust the weight.
    :  cumreward = cumreward + fabs(reinf)
    :} : cumulative reward magnitude; only if weight changed
  }
}

FUNCTION hebbRL () { : RL from pre before post spiking
  :if (NOSTDPTAG) {
  :  hebbRL = RLhebbwt
  :}
  :else
  if (tlasthebbelig < 0.0) { : If eligibility has not occurred yet return 0.0.
    hebbRL = 0.0
  }
  else if (useRLexp == 0) { : If we are using a binary (i.e. square-wave) eligibility traces...
    if (t - tlasthebbelig <= RLlenhebb) { : If we are within the length of the eligibility trace...
      hebbRL = RLhebbwt : Otherwise (outside the length), return 0.0.
    }
    else {
      hebbRL = 0.0
    }
  }
  else {  : Otherwise (if we are using an exponential decay traces)...use the Hebbian decay to calculate the gain.
    hebbRL = RLhebbwt * exp((tlasthebbelig - t) / RLlenhebb)
  }
}

FUNCTION antiRL () { : RL from post before pre spiking
  :if (NOSTDPTAG) {
  :  antiRL = RLantiwt
  :}
  if (tlastantielig < 0.0) { : If eligibility has not occurred yet return 0.0.
    antiRL = 0.0
  }
  else if (useRLexp == 0) { : If we are using a binary (i.e. square-wave) eligibility traces...
    if (t - tlastantielig <= RLlenanti) { : If we are within the length of the eligibility trace...
      antiRL = RLantiwt
    }
    else { : Otherwise (outside the length), return 0.0.
      antiRL = 0.0
    }
  }
  else { : Otherwise (if we are using an exponential decay traces), use the anti-Hebbian decay to calculate the gain.
    antiRL = RLantiwt * exp((tlastantielig - t) / RLlenanti)
  }
}

FUNCTION softthreshold (rawwc) {
  if (rawwc >= 0) {
    softthreshold = rawwc * (1.0 - synweight / wmax) : If the weight change is non-negative, scale by 1 - weight / wmax.
  }
  else { : Otherwise (the weight change is negative), scale by weight / wmax.
    softthreshold = rawwc * synweight / wmax
  }
}

PROCEDURE adjustweight (wc) {
   synweight = synweight + wc : apply the synaptic modification, and then clip the weight if necessary to make sure it is between wbase and wmax.
   if (synweight > wmax) { synweight = wmax }
   if (synweight < wbase) { synweight = wbase }
}
