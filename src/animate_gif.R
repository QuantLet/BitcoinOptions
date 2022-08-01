#Script that calculates all the centralities for the FRM Crypto Data and creates Movies of the plots
setwd('/Users/julian/src/spd')
#Check if qgraph is installed, if not: install, either way: load qgraph
if (!require('ggplot2')) install.packages('ggplot2'); library('ggplot2')
if (!require('gganimate')) install.packages('gganimate'); library('gganimate')
if (!require('av')) install.packages('av'); library('av')
if (!require('gifski')) install.packages('gifski'); library('gifski')
if (!require('strex')) install.packages('strex'); library('strex')
library('data.table')
library('plotly')

options(gganimate.dev_args = list(bg = 'transparent'))

out_dir = 'out/3dplots'
data_dir = 'out/movies'
f = list.files(data_dir, pattern = 'btc_pk_',full.names = TRUE)

dat = lapply(f, function(z){

    d = read.csv(z)

    # Decompose filename
    # 1) Path
    # 2) PK
    # 3) Tau
    # 4) Date
    splt = strsplit(z, '_')[[1]]
    d$tau = as.numeric(splt[3])
    d$date = as.Date(substr(splt[4], 1, 10))
    d$X = NULL

    return(d)
}) # Cols: m, spdy, cb_up, cb_dn, pk

bound = do.call('rbind.data.frame', dat)
setDT(bound)

# Kick Outliers
rm_outliers = function(x){
    x[x %in% boxplot.stats(x)$out] = NA
    return(x)
}

dd = apply(bound[,-7], 2, rm_outliers)
dt = data.table(dd)
dt$date = bound$date
dt = na.omit(dt)

dt$tau_weeks = round(dt$tau * 365 / 7) # in Weeks


gen_gif = function(.bound, .tau){

    d = .bound[.tau == tau]


    #Set ggplot2 theme
    theme_set(theme_bw())
    plot1 <- ggplot(d, 
                    aes(x = m,
                        y = pk, 
                        colour = date, group = 1)) + # identifier of observations across states
      #geom_jitter(alpha = 0.3, # to reduce overplotting: jitttering & alpha
      #width = 5) + 
      #geom_line() #+ 
      geom_point()
    labs(y = 'pk', # no axis label
        x = 'm',  # x-axis label
        title = "Clustering") +
      theme_minimal() +  # apply minimal theme
      theme(panel.grid = element_blank(),  # remove all lines of plot raster
            text = element_text(size = 16)) # increase font size

    ## Animated Plot
    anim <- plot1 + 
      transition_reveal(date, keep_last = T) #

    animate(anim, width = 600, height = 550, renderer = gifski_renderer(), bg = 'transparent')
    anim_save(paste0(data_dir, '/pricingkernels_over_time_', .tau, '.gif'), transparent = TRUE)

    return(TRUE)

}


# Activate for Movie Generation
# unique_taus = unique(bound$tau)
#for(t in unique_taus){
#  gen_gif(dt, t)
#}


gen_3dplot = function(.dt, day, out){

  #d = .bound[tau_weeks == .tau]
  d = .dt[date == day][order(m)]
  fig = plot_ly(d, x = ~m, y = ~tau_weeks, z = ~pk, type = 'scatter3d', marker = list(size = 2))  %>% layout(scene = list(xaxis = list(title = 'Moneyness'), yaxis = list(title = 'Weeks to Maturity'), zaxis = list(title = 'Pricing Kernel')))
  fig
  fn = paste0(out, '/3dplots/3dpricingkernels_', as.Date(day, origin = '1970-01-01'), '.png')
  orca(p = fig, file = fn)

  return(fig)

}

unique_days = unique(dt$date)

for(currday in unique_days){
  gen_3dplot(dt, currday, out_dir)
}

