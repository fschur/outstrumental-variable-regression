# Schwartz-style Poisson reference for PM2.5 increasing from 10 to 20 µg/m³.
suppressPackageStartupMessages(library(gam))
panel <- read.csv(commandArgs(trailingOnly = TRUE)[1])
panel$date <- as.Date(panel$date)
panel$day <- as.integer(panel$date - min(panel$date))
panel$weekday <- factor(format(panel$date, "%u"))
panel$rain <- as.integer(as.logical(panel$rain))
panel$snow <- as.integer(as.logical(panel$snow))
panel$pm10unit <- panel$pm25 / 10

predict_city <- function(d) {
    fit <- gam(death ~ pm10unit + weekday + rain + snow + lo(day, span = .05) +
                   lo(air_temp, span = .50) + lo(dewpoint, span = .50),
               data = d, family = poisson(), na.action = na.fail,
               control = gam.control(epsilon = 1e-8, bf.epsilon = 1e-8, maxit = 100, bf.maxit = 300))
    beta <- unname(coef(fit)["pm10unit"])
    # Change only PM2.5 in each day's fitted mean, then convert counts to deaths per million.
    rate_at_10 <- 1e6 * fitted(fit) / d$population * exp(beta * (1 - d$pm10unit))
    c(baseline = sum(rate_at_10), change = sum(rate_at_10) * expm1(beta))
}

totals <- rowSums(vapply(split(panel, panel$city), predict_city, c(baseline = 0, change = 0)))
absolute <- totals["change"] / nrow(panel)
percent <- 100 * totals["change"] / totals["baseline"]
cat(sprintf("%.15g %.15g\n", absolute, percent))
