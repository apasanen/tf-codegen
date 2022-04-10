TRAIN=train.csv
TEST=test.csv
MODEL=model
PYTHON=python
LITE=model.tflite
LITEQ=model-quant.tflite
LITEQ2=model-quant2.tflite

all:result.txt head.csv

$(TRAIN):
	$(PYTHON) genData.py --fig data.pdf $(TRAIN) $(TEST)

head.csv:$(TEST)
	head -n 3 $< > $@

%.tflite:%
	$(PYTHON) tf2tflite.py $^ $@
	cp -v $@ /mnt/c/Users/antti

%-quant.tflite:%
	$(PYTHON) tf2tflite.py --quant $(TRAIN) $^ $@
	cp -v $@ /mnt/c/Users/antti

%-quant2.tflite:%
	$(PYTHON) tf2tflite.py --ioint --quant  $(TRAIN) $^ $@
	cp -v $@ /mnt/c/Users/antti

$(TEST):$(TRAIN)

$(MODEL):$(TRAIN)
	$(PYTHON) gen-model.py $< $@

out-0.txt:$(TEST) $(MODEL)
	$(PYTHON) evaluate.py $^ $@

out-1.txt:$(TEST) $(MODEL)
	$(PYTHON) evaluate.py -c $^ $@

out-2.txt:$(TEST) $(LITE)
	$(PYTHON) evaluate.py $^ $@

out-3.txt:$(TEST) $(LITE)
	$(PYTHON) evaluate.py -c $^ $@

out-4.txt:$(TEST) $(LITEQ)
	$(PYTHON) evaluate.py $^ $@

out-5.txt:$(TEST) $(LITEQ2)
	$(PYTHON) evaluate.py $^ $@

result.txt:out-0.txt out-1.txt out-2.txt out-3.txt out-4.txt out-5.txt 
	$(PYTHON) check.py $^

clean:
	$(RM) -r $(MODEL) $(TRAIN) $(TEST) $(RES)
