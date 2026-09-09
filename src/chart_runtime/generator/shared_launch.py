"""WHERE mirrors shared launch actions; only independent inputs spend free hands.

A same-head instantaneous outer hit shares the existing launch action. Hold
does not: it must remain pressed while the Slide moves away. Actual trajectory,
Touch grouping and all other interaction verdicts remain in the CUDA Harness.
"""


class HandAccounting:
    """The sole WHERE authority for launch sharing and independent hands."""
    @staticmethod
    def launch_heads(snapshot):
        lanes=set(snapshot.get('launchShareLanes',()))
        if snapshot.get('_launch_cue_lane') is not None:lanes.add(int(snapshot['_launch_cue_lane']))
        return lanes

    def maximum_credit(self,snapshot,families):
        return min(len(self.launch_heads(snapshot)),sum(f==0 for f in families))

    def independent_outer(self,snapshot,starts,families):
        lanes=self.launch_heads(snapshot)
        shared=sum(int(lane) in lanes and f==0 for lane,f in zip(starts,families))
        return len(families)-shared

    def account(self,snapshot,starts,families,touch_groups=0):
        available=int(snapshot.get('holdAvailableHands',2));independent=self.independent_outer(snapshot,starts,families)
        baseline=max(0,2-available);total=baseline+independent+int(touch_groups)
        return {'activeSlideHands':int(snapshot.get('activeSlideHands',0)),'baselineHands':baseline,
                'sharedLaunchTaps':len(families)-independent,'independentOuterInputs':independent,
                'touchGroups':int(touch_groups),'totalHands':total,'availableHands':max(0,2-total),'fits':total<=2}

    def remaining_hands(self,snapshot,starts,families):
        return self.account(snapshot,starts,families)['availableHands']

    def assignment_fits(self,snapshot,starts,families):
        return self.account(snapshot,starts,families)['fits']


HAND_ACCOUNTING=HandAccounting()
