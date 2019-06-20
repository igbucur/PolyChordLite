# Additional libraries
LDLIBS += -larmadillo -lconfig++

# rule for building the c++ wrapper
$(LIB_DIR)/libMR_likelihood.a: MR_likelihood.o
	$(AR) $@ $^ 

# General rule for building object file (.o)  from c++ files (.cpp)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(POLYCHORD_DIR) -c $<
# General rule for building object file (.o)  from c files (.c)
%.o: %.c
	$(CC) $(CFLAGS) -I$(POLYCHORD_DIR) -c $<

# Utility targets
.PHONY: clean veryclean

clean:
	$(RM) *.o *.mod *.MOD

veryclean: clean
	$(RM) *~ *.a
