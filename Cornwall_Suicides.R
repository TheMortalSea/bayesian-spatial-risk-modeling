library("sf")
library("tmap")
library("spdep")
library("rstan")
library("geostan")
library("SpatialEpi")
library("tidybayes")
library("tidyverse")
library("knitr")

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#######################################################
# set working directory and load/join all files

setwd("filepath")

# load in the LSOA geopackage for the spatial reference
cornwall_lsoa_shp <- read_sf("Cornwall_LSOA.gpkg")
cornwall_lsoa_shp <- cornwall_lsoa_shp %>% select(-LSOA21NMW)

# load in the suicide per LSOA count data
suicides <- read.csv("suicide_by_lsoa.csv")
nrow(suicides)
# load in the IMD data by LSOA
IMD_data <- read.csv("imd2019lsoa.csv")
nrow(IMD_data)

# load in the population count per LSOA data
population <- read.csv("LSOA_POP.csv")
nrow(population)

samhi_data <- read.csv("SAMHI_Index.csv")

#######################################################
# clean and combine deprivation data, population data, and the suicide data (other covariates can be added later if needed and are justified in research)
# CLEAN IMD DATA
names(IMD_data) <- c("LSOA", "Year", "MeasureType", "NA_col", "Value", "Domain")
IMD_data <- IMD_data %>% select(LSOA, Year, MeasureType, Value, Domain)

IMD_data$Domain <- gsub("^[a-zA-Z]\\.\\s*", "", IMD_data$Domain)

IMD_data <- IMD_data %>%
  mutate(Domain_Measure = paste0(gsub("[^A-Za-z0-9]", "", Domain), "_", MeasureType))
nrow(IMD_data)

IMD_data <- IMD_data %>%
  filter(MeasureType == "Score")
nrow(IMD_data)

IMD_data <- IMD_data %>%
  select(LSOA, Year, Domain_Measure, Value) %>%
  pivot_wider(names_from = Domain_Measure, values_from = Value)
nrow(IMD_data)

names(IMD_data)[names(IMD_data) == "IndexofMultipleDeprivationIMD_Score"] <- "IMD_Score"
#######################################################
# COMBINE POPULATION AND SUICIDE DATA
suicides$LSOA <- sub(".*?(E[0-9]+).*", "\\1", suicides$LSOA)

names(population)[names(population) == "Total"] <- "Population_Total"
names(population)[names(population) == "LAD.2021.Name"] <- "Name"
names(population)[names(population) == "LSOA.2021.Code"] <- "LSOA"

suicides$LSOA <- as.character(suicides$LSOA)
population$LSOA <- as.character(population$LSOA)

suicides$LSOA <- trimws(suicides$LSOA)
population$LSOA <- trimws(population$LSOA)

head(suicides$LSOA)
head(population$LSOA)

suicides <- population %>%
  inner_join(suicides, by = "LSOA")
nrow(suicides)

suicides <- IMD_data %>%
  inner_join(suicides, by = "LSOA")
nrow(suicides)

suicides <- samhi_data %>%
  inner_join(suicides, by = "LSOA")
nrow(suicides)

suicides <- suicides %>% select(Name, LSOA, Population_Total, Deaths, Suicides, IMD_Score, samhi_index)
str(suicides)

mean_value <- mean(suicides$Suicides, na.rm = TRUE)
print(mean_value)
var_value <- var(suicides$Suicides, na.rm = TRUE)
print(var_value)

hist(suicides$Suicides, main = "Histogram of Suicides", xlab = "Suicides", col = "skyblue", border = "black")

# Plot a boxplot to visually check for skewness
boxplot(suicides$Suicides, main = "Boxplot of Suicides", ylab = "Suicides", col = "lightgreen")


suicides$Population_Total <- as.numeric(gsub(",", "", suicides$Population_Total))
suicides$Deaths <- as.numeric(suicides$Deaths)

#######################################################
# calculate the expected number (expected rate * population for each LSOA)
suicides$ExpectedNum <- expected(population = suicides$Population_Total, cases = suicides$Suicides, n.strata = 1)

#######################################################
# merge the attribute table to the shapefile
spatial.data <- merge(cornwall_lsoa_shp, suicides, by.x = c("LSOA21CD"), by.y = c("LSOA"), all.x = TRUE)
nrow(spatial.data)

missing_lsoa <- spatial.data$LSOA21CD[is.na(spatial.data$Suicides)]
print(missing_lsoa)

spatial.data <- na.omit(spatial.data)
nrow(spatial.data)
# reordering the columns
#spatial.data <- spatial.data[, c(3,1,2,4,5,7,6)]

# need to be coerced into a spatial object
sp.object <- as(spatial.data, "Spatial")

# needs to be coerced into a matrix object
adjacencyMatrix <- shape2mat(sp.object)

# we extract the components for the ICAR model
extractComponents <- prep_icar_data(adjacencyMatrix)

n <- as.numeric(extractComponents$group_size)
nod1 <- extractComponents$node1
nod2 <- extractComponents$node2
n_edges <- as.numeric(extractComponents$n_edges)

y <- spatial.data$Suicides
x <- as.numeric(scale(spatial.data$IMD_Score))
x2 <- as.numeric(scale(spatial.data$samhi_index))
e <- spatial.data$ExpectedNum

# put all components into a list object
stan.spatial.dataset <- list(
  N = n,
  N_edges = n_edges,
  node1 = nod1,
  node2 = nod2,
  Y = y,
  X = x,       # First covariate (IMD_Score)
  X2 = x2,     # Second covariate (samhi_index)
  Off_set = e
)
colSums(is.na(suicides))

colSums(is.na(spatial.data))

str(stan.spatial.dataset)

#stan.spatial.dataset <- as.data.frame(stan.spatial.dataset)  # In case it's not a data frame
#colSums(is.na(stan.spatial.dataset))
# old iter was 20000
zero_inflated = stan("NegBinomial_test.stan", data=stan.spatial.dataset, iter=15000, control = list(max_treedepth = 12), chains=10, verbose = FALSE)
#neg_binomial_fit = stan("neg_nozero.stan", data=stan.spatial.dataset, iter=15000, control = list(max_treedepth = 12), chains=10, verbose = FALSE)
#neg_log = stan("neg_2_log.stan", data=stan.spatial.dataset, iter=15000, control = list(max_treedepth = 12), chains=10, verbose = FALSE)
#zero_poisson = stan("zero_poisson.stan", data=stan.spatial.dataset, iter=15000, control = list(max_treedepth = 12), chains=10, verbose = FALSE)
#######################################################
#Compare the models
library("loo")
loo_result <- loo(neg_log)
print(loo_result)

loo_result2 <- loo(neg_binomial_fit)
print(loo_result2)

loo_result3 <- loo(zero_inflated)
print(loo_result3)

loo_result4 <- loo(zero_poisson)
print(loo_result4)

#######################################################
# Extract the summary statistics directly from the Stan model
summary_fit <- summary(zero_inflated)
posterior_summary <- summary_fit$summary

# Define the key parameters you want in your table
key_params <- c("alpha", "beta", "beta2", "sigma", "rho")

# If you're using the ZINB model, include these parameters as well
if ("phi_nb" %in% rownames(posterior_summary)) {
  key_params <- c(key_params, "phi_nb")
}
if ("gamma_0" %in% rownames(posterior_summary)) {
  key_params <- c(key_params, "gamma_0", "gamma_1")
}
if ("gamma_2" %in% rownames(posterior_summary)) {
  key_params <- c(key_params, "gamma_2")  # Include gamma_2 if it's in the model
}

# Extract the rows for these parameters
table_data <- posterior_summary[key_params, c("mean", "se_mean", "sd", "2.5%", "97.5%", "n_eff", "Rhat")]

# Format the table with a simple print statement
print(round(table_data, 6))

#######################################################
# create categories to define if an area has significant increase or decrease in risk, or nothing all 
spatial.data$Significance <- NA
spatial.data$Significance[spatial.data$rrlower<1 & spatial.data$rrupper>1] <- 0    # NOT SIGNIFICANT
spatial.data$Significance[spatial.data$rrlower==1 | spatial.data$rrupper==1] <- 0  # NOT SIGNIFICANT
spatial.data$Significance[spatial.data$rrlower>1 & spatial.data$rrupper>1] <- 1    # SIGNIFICANT INCREASE
spatial.data$Significance[spatial.data$rrlower<1 & spatial.data$rrupper<1] <- -1   # SIGNIFICANT DECREASE

relativeRisk.results <- as.data.frame(summary(model, pars=c("rr_mu"), probs=c(0.025, 0.975))$summary)
# now cleaning up this table up
# first, insert clean row numbers to new data frame
row.names(relativeRisk.results) <- 1:nrow(relativeRisk.results)
# second, rearrange the columns into order
relativeRisk.results <- relativeRisk.results[, c(1,4,5,7)]
# third, rename the columns appropriately
colnames(relativeRisk.results)[1] <- "rr"
colnames(relativeRisk.results)[2] <- "rrlower"
colnames(relativeRisk.results)[3] <- "rrupper"
colnames(relativeRisk.results)[4] <- "rHAT"

# view clean table 
head(relativeRisk.results)

spatial.data$rr <- relativeRisk.results[, "rr"]
spatial.data$rrlower <- relativeRisk.results[, "rrlower"]
spatial.data$rrupper <- relativeRisk.results[, "rrupper"]

# creating the labels
RiskCategorylist <- c(">0.0 to 0.25", "0.26 to 0.50", "0.51 to 0.75", "0.76 to 0.99", "1.00 & <1.01",
                      "1.01 to 1.10", "1.11 to 1.25", "1.26 to 1.50", "1.51 to 1.75", "1.76 to 2.00", "2.01 to 3.00")

# next, we are creating the discrete colour changes for my legends and want to use a divergent colour scheme
# scheme ranges from extreme dark blues to light blues to white to light reds to extreme dark reds
# you can pick your own colour choices by checking out this link [https://colorbrewer2.org]

RRPalette <- c("#65bafe","#98cffe","#cbe6fe","#dfeffe","white","#fed5d5","#fcbba1","#fc9272","#fb6a4a","#de2d26","#a50f15")

# categorising the risk values to match the labelling in RiskCategorylist object
spatial.data$RelativeRiskCat <- NA
spatial.data$RelativeRiskCat[spatial.data$rr>= 0 & spatial.data$rr <= 0.25] <- -4
spatial.data$RelativeRiskCat[spatial.data$rr> 0.25 & spatial.data$rr <= 0.50] <- -3
spatial.data$RelativeRiskCat[spatial.data$rr> 0.50 & spatial.data$rr <= 0.75] <- -2
spatial.data$RelativeRiskCat[spatial.data$rr> 0.75 & spatial.data$rr < 1] <- -1
spatial.data$RelativeRiskCat[spatial.data$rr>= 1.00 & spatial.data$rr < 1.01] <- 0
spatial.data$RelativeRiskCat[spatial.data$rr>= 1.01 & spatial.data$rr <= 1.10] <- 1
spatial.data$RelativeRiskCat[spatial.data$rr> 1.10 & spatial.data$rr <= 1.25] <- 2
spatial.data$RelativeRiskCat[spatial.data$rr> 1.25 & spatial.data$rr <= 1.50] <- 3
spatial.data$RelativeRiskCat[spatial.data$rr> 1.50 & spatial.data$rr <= 1.75] <- 4
spatial.data$RelativeRiskCat[spatial.data$rr> 1.75 & spatial.data$rr <= 2.00] <- 5
spatial.data$RelativeRiskCat[spatial.data$rr> 2.00 & spatial.data$rr <= 10] <- 6

# check to see if legend scheme is balanced - if a number is missing that categorisation is wrong!
table(spatial.data$RelativeRiskCat)

# map of relative risk
rr_map <- tm_shape(spatial.data) + 
  tm_fill("RelativeRiskCat", style = "cat", title = "Relavtive Risk", palette = RRPalette, labels = RiskCategorylist) +
  tm_shape(cornwall_lsoa_shp) + tm_polygons(alpha = 0.05) + 
  tm_layout(frame = FALSE, legend.outside = TRUE, legend.title.size = 0.8, legend.text.size = 0.7) +
  tm_compass(position = c("right", "top"), size = 1.5) + tm_scale_bar(position = c("right", "bottom"))

rr_map

# map of significance regions
sg_map <- tm_shape(spatial.data) + 
  tm_fill("Significance", style = "cat", title = "Significance Categories", 
          palette = c("#33a6fe", "white", "#fe0000"), labels = c("Significantly low", "Not Significant", "Significantly high")) +
  tm_shape(cornwall_lsoa_shp) + tm_polygons(alpha = 0.10) +
  tm_layout(frame = FALSE, legend.outside = TRUE, legend.title.size = 0.8, legend.text.size = 0.7) +
  tm_compass(position = c("right", "top"), size = 1.5) + tm_scale_bar(position = c("right", "bottom"))

sg_map

# create side-by-side plot
tmap_arrange(rr_map, sg_map, ncol = 2, nrow = 1)

summary(spatial.data$rrlower)
summary(spatial.data$rrupper)

# extract the exceedence probabilities from the icar_possion_fit object
# compute the probability that an area has a relative risk ratio > 1.0
threshold <- function(x){mean(x > 1.00)}
excProbrr <- model %>% spread_draws(rr_mu[i]) %>% 
  group_by(i) %>% summarise(rr_mu=threshold(rr_mu)) %>%
  pull(rr_mu)

# insert the exceedance values into the spatial data frame
spatial.data$excProb <- excProbrr

# create the labels for the probabilities
ProbCategorylist <- c("<0.01", "0.01-0.09", "0.10-0.19", "0.20-0.29", "0.30-0.39", "0.40-0.49","0.50-0.59", "0.60-0.69", "0.70-0.79", "0.80-0.89", "0.90-0.99", "1.00")

# categorising the probabilities in bands of 10s
spatial.data$ProbCat <- NA
spatial.data$ProbCat[spatial.data$excProb>=0 & spatial.data$excProb< 0.01] <- 1
spatial.data$ProbCat[spatial.data$excProb>=0.01 & spatial.data$excProb< 0.10] <- 2
spatial.data$ProbCat[spatial.data$excProb>=0.10 & spatial.data$excProb< 0.20] <- 3
spatial.data$ProbCat[spatial.data$excProb>=0.20 & spatial.data$excProb< 0.30] <- 4
spatial.data$ProbCat[spatial.data$excProb>=0.30 & spatial.data$excProb< 0.40] <- 5
spatial.data$ProbCat[spatial.data$excProb>=0.40 & spatial.data$excProb< 0.50] <- 6
spatial.data$ProbCat[spatial.data$excProb>=0.50 & spatial.data$excProb< 0.60] <- 7
spatial.data$ProbCat[spatial.data$excProb>=0.60 & spatial.data$excProb< 0.70] <- 8
spatial.data$ProbCat[spatial.data$excProb>=0.70 & spatial.data$excProb< 0.80] <- 9
spatial.data$ProbCat[spatial.data$excProb>=0.80 & spatial.data$excProb< 0.90] <- 10
spatial.data$ProbCat[spatial.data$excProb>=0.90 & spatial.data$excProb< 1.00] <- 11
spatial.data$ProbCat[spatial.data$excProb == 1.00] <- 12

# check to see if legend scheme is balanced
table(spatial.data$ProbCat)

hist(spatial.data$excProb, 
     breaks = 20,      # Number of bars (you can adjust)
     main = "Histogram of Exceedance Probabilities", 
     xlab = "Exceedance Probability (P(RR > 1))", 
     ylab = "Number of Areas",
     col = "skyblue", 
     border = "white")

# map of exceedance probabilities
tm_shape(spatial.data) + 
  tm_fill("ProbCat", style = "cat", title = "Probability", palette = "GnBu", labels = ProbCategorylist) +
  tm_shape(cornwall_lsoa_shp) + tm_polygons(alpha = 0.05, border.col = NA) +  # Change border thickness here
  tm_layout(frame = FALSE, legend.outside = TRUE, legend.title.size = 0.8, legend.text.size = 0.7) +
  tm_compass(position = c("right", "top"), size = 1.5) + tm_scale_bar(position = c("right", "bottom"))

