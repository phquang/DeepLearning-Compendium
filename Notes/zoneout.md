## To be updated
This paper presents a way to reguarlize RNNs during training.
Previous work such as [] showed that directly appling Dropout [] to the recurrent connection in RNN may hurt its performance.
This paper suggests that during training, instead of randomly dropping the value of a cell (dropout), a cell is randomly not updated when a new input arrives.
Another way to connect Zoneout with Dropout is that in Dropout, cell's value is randomly masked with mask 0 while the mask value is its value at the previous step.
(This explanation is similar to masking the transition function with 0 in Dropout and 1 in Zoneout as stated in the paper)
During testing, both methods  behave similarly, the cell value is multiplied by the expectation of the Dropout/Zoneout probability.
