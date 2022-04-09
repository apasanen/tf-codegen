TRAIN=train.csv
TEST=test.csv
MODEL=model.h5
PYTHON=python

all:$(MODEL)

$(TRAIN):
	$(PYTHON) gen-data.py $(TRAIN) $(TEST)

$(TEST):$(TRAIN)

$(MODEL):$(TRAIN)
	$(PYTHON) gen-model.py $< $@

res1.txt:$(MODEL) $(TEST)
	$(PYTHON) evaluate.py $<

clean:
	$(RM) $(MODEL) $(TRAIN) $(TEST)
