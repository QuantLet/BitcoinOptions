
# install and load packages
libraries = c("KernSmooth", "matlab", "kedd", "scatterplot3d", "scales", "RColorBrewer", "data.table") #  "rgl"
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x, dependencies = TRUE)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# ---------------------------------------------------------------------------#
# Data preparation                                                           #
# ---------------------------------------------------------------------------#

wd = paste0(Sys.getenv('HOME'), '/source/repos/spd/')
setwd(wd)

source('pricingkernel/calc.R')
source('pricingkernel/utils.R')
#source('pricingkernel/poly.R')

unique_only = TRUE
do_kick_outliers = TRUE
warning(paste0('unique only: ', unique_only))
#load('data/orderbook.RData', verbose = TRUE)
#obs = z[[1]]
#obs$asks = NULL
#obs$bids = NULL
#write.csv(obs, file = 'data/ob_params.csv')
#fwrite(obs, file = 'data/orderbooks_test.csv')
#rc = read_and_clean()
rc = read_and_clean_orderbooks()
odata_all = rc$odax
idata = rc$idata
rm(rc)

unique_maturities = sort(unique(odata_all[,8]))

#setDT(odata_all)
for(maturity in unique_maturities){

  print(as.Date(maturity, origin = '1970-01-01'))
  odax = restrict_data(odata_all, target_date = maturity, make_unique = unique_only, kick_outliers = do_kick_outliers)


  # ---------------------------------------------------------------------------#
  # Estimation and plotting of Risk Neutral Densities                          #
  # ---------------------------------------------------------------------------#
  rounded_maturities = round(odax$maturity, 2)
  dtau         = round(rounded_maturities * 365)  # maturity in days
  odax         = cbind(odax, dtau)
  mat          = odax[, 4]
  IR           = odax[, 3]
  ForwardPrice = odax[, 1] * exp(IR * mat)
  RawData      = cbind(ForwardPrice, odax[, 2], IR, mat, odax[, 5], odax[, 6], 
                      odax[, 7], odax[, 9])



  ####################### Plot SPD for different maturities #############################

  tau_day = sort(unique(RawData[,8]))
  SPD     = list()

  col_vector = generate_colors()
  col = col_vector[1:length(tau_day)]


  # Plot State-price Densities together
  if(unique_only){
    png_fname = paste0('pricingkernel/plots/unique/Risk-neutral density for different Maturities: ', as.Date(maturity, origin = '1970-01-01'),'.png')
  }else{
    png_fname = paste0('pricingkernel/plots/nonunique/Risk-neutral density for different Maturities: ', as.Date(maturity, origin = '1970-01-01'),'.png')
  }
  
  for (i in 1:length(tau_day)) {
  test_f = try(SPDrookley(RawData, tau_day[i]))
  if(!'try-error' %in% class(test_f)){
    SPD[[i]] = test_f
    }
  }
    

  png(png_fname)
  par(xpd = TRUE)
  legend_labels = paste(as.character(tau_day), "Days")
  plot(SPD[[1]]$xGrid, SPD[[1]]$result, xlim = c(7200, 10000), ylim = c(0, 
      8e-03), xlab = "Spot Price", ylab = "Density",
      main = "Risk neutral density for different maturities", col = col[1], type = 'l') # , lwd = 3

  for(i in 2:length(SPD)){
    lines(SPD[[i]]$xGrid, SPD[[i]]$result, col = col[i]) #, lwd = 3
  }
  legend("topright", legend_labels, lty = 1, lwd = 2, col = col, bty = "n", cex = 0.8)
  dev.off()

  browser()
}
